mod ekf;
use core::f64;
use ekf::{ExtendedKalmanFilter, MeasurementModel, SystemModel};
use ekf_server::RerunHandler;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
//use rerun::external::arrow::csv::reader;
use rerun::{RecordingStream, RecordingStreamBuilder};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use ekf::wrap_to_pi;

/// Shared EKF state across all connections
struct SharedEkfState {
    ekf: ExtendedKalmanFilter,
    last_update_client: Option<usize>,
    update_count: usize,
}

impl SharedEkfState {
    fn new() -> Self {
        // Initialize state at (0.247, 0.635)
        let initial_state = Vector3::new(0.0, 0.0, 0.0);
        let initial_covariance = Matrix3::identity() * 0.01;
        let ekf = ExtendedKalmanFilter::new(initial_state, initial_covariance);

        Self {
            ekf,
            last_update_client: None,
            update_count: 0,
        }
    }
}

/// Parse incoming message format: timestamp,h,j,theta,d,del_t
#[derive(Debug)]
struct MeasurementData {
    timestamp: i64,
    h: f64,
    j: f64,
    theta: f64,
    d: f64,
    del_t: f64,
}

impl MeasurementData {
    fn parse(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() != 6 {
            return None;
        }

        Some(Self {
            timestamp: parts[0].parse().ok()?,
            h: parts[1].parse().ok()?,
            j: parts[2].parse().ok()?,
            theta: parts[3].parse().ok()?,
            d: parts[4].parse().ok()?,
            del_t: parts[5].parse().ok()?,
        })
    }
}

/// Handle individual client connection
fn handle_client(
    stream: TcpStream,
    shared_info: Arc<Mutex<(SharedEkfState, Vector2<f64>)>>,
    client_id: usize,
    mean_sender: mpsc::Sender<(f32, f32)>,
    cov_sender: mpsc::Sender<Matrix2<f32>>,
) {
    let peer_addr = stream.peer_addr().unwrap();
    println!("[Client {}] Connected from: {}", client_id, peer_addr);

    let mut reader = BufReader::new(stream);
    println!("STREAM STARTED");

    loop {
        let mut buf = String::new();
        if let Err(e) = reader.read_line(&mut buf) {
            println!("ERROR: {:?}", e);
            continue;
        }

        let buf = buf.trim().to_string();
        let data = match MeasurementData::parse(buf.as_str()) {
            Some(val) => val,
            None => continue,
        };

        if 340.0 * data.del_t > data.d {
            continue;
        }

        let angle = wrap_to_pi((340.0 * data.del_t / data.d).asin());

        // Lock the mutex and get a mutable reference to the inner tuple
        let mut guard = shared_info.lock().unwrap();
        let (ref mut state, ref mut control) = *guard; // borrow, not move

        // Check if this is a different client than the last update
        let should_update = match state.last_update_client {
            None => true,
            Some(last_client) => last_client != client_id,
        };

        if !should_update {
            continue;
        }

        let measurement_model = MeasurementModel::new(data.h, data.j, data.theta, 0.01);
        let system_model = SystemModel::new(0.01);

        state.ekf.predict(&system_model, control);
        state.ekf.update(&measurement_model, angle);
        state.last_update_client = Some(client_id);
        state.update_count += 1;

        let position = state.ekf.get_state();
        let covariance = state.ekf.get_covariance();

        println!(
            "[Client {}] Position: {} , {}",
            client_id, position[0], position[1]
        );
        println!(
            "[Client {}] Covariance: {} , {}",
            client_id,
            covariance[(0, 0)],
            covariance[(1, 1)]
        );

        let covariance_2d: Matrix2<f32> = Matrix2::new(
            covariance[(0, 0)] as f32,
            covariance[(0, 1)] as f32,
            covariance[(1, 0)] as f32,
            covariance[(1, 1)] as f32,
        );

        let _ = cov_sender.send(covariance_2d);
        let _ = mean_sender.send((position[0] as f32, position[1] as f32));
    }
}

fn handle_controller_connection(
    stream: TcpStream,
    shared_info: Arc<Mutex<(SharedEkfState, Vector2<f64>)>>,
) -> Result<(), ()> {
    stream.set_nodelay(true).unwrap();
    println!("Controller connection started");

    // Wrap the stream in a BufReader that owns it.
    let mut reader = BufReader::new(stream);
    loop {
        let mut buf = String::new();
        if let Err(e) = reader.read_line(&mut buf) {
            println!("ERROR: {:?}", e);
            continue;
        }

        let buf = buf.trim().to_string();
        let numbers: Vec<String> = buf.split(',').map(String::from).collect();
        if numbers.len() < 2 {
            continue;
        }

        let v = numbers[0].parse().unwrap_or(0.0);
        let omega = numbers[1].parse().unwrap_or(0.0);

        // Lock mutex and update control
        let mut guard = shared_info.lock().unwrap();
        let (ref mut state, ref mut control) = *guard;
        *control = Vector2::new(v, omega);

        dbg!(control);

        let pos = state.ekf.get_state();
        let response = format!("{},{},{}\n", pos[0], pos[1], pos[2]);

        // Write using the underlying stream (BufReader gives mutable access)
        if let Err(e) = reader.get_mut().write_all(response.as_bytes()) {
            println!("Write error: {:?}", e);
        }
    }
}

fn main() -> std::io::Result<()> {
    let addr = "10.220.135.191:6060";
    let full_info_mutex = Arc::new(Mutex::new((
        SharedEkfState::new(),
        Vector2::<f64>::new(0.0, 0.0),
    )));

    // Initial prediction
    {
        let mut guard = full_info_mutex.lock().unwrap();
        let (ref mut state, ref mut control) = *guard;
        let system_model = SystemModel::new(0.01);
        state.ekf.predict(&system_model, control);
        println!("EKF initialized at: {:?}", state.ekf.get_state());
    }

    let listener = TcpListener::bind(addr)?;
    println!("Server listening on: {}", addr);

    let (tx_mean, rx_mean) = mpsc::channel();
    let (tx_cov, rx_cov) = mpsc::channel();

    // Visualization thread
    thread::spawn(move || {
        let rec = RecordingStreamBuilder::new("ekf_visualization")
            .spawn()
            .unwrap();
        let rerun_handler =
            RerunHandler::new(rec, "ExtendedKalmanfilter".to_string(), rx_mean, rx_cov);
        rerun_handler.run();
    });

    // Controller listener thread
    let cloned_info = Arc::clone(&full_info_mutex);
    thread::spawn(move || {
        let addr_input = "10.220.135.191:9100";
        let control_listener = TcpListener::bind(addr_input).unwrap();
        println!("Controller listening on: {}", addr_input);
        for stream in control_listener.incoming() {
            if let Ok(stream) = stream {
                let _ = handle_controller_connection(stream, cloned_info.clone());
            }
        }
    });

    // Client connections
    let mut client_counter = 0;
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                client_counter += 1;
                let full_info_clone = Arc::clone(&full_info_mutex);
                let tx_mean = tx_mean.clone();
                let tx_cov = tx_cov.clone();
                thread::spawn(move || {
                    handle_client(stream, full_info_clone, client_counter, tx_mean, tx_cov);
                });
            }
            Err(e) => eprintln!("Error accepting connection: {}", e),
        }
    }

    Ok(())
}
