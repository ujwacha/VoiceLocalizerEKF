mod ekf;
use core::f64;
use ekf::{ExtendedKalmanFilter, MeasurementModel, SystemModel};
use ekf_server::RerunHandler;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
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
    shared_state: Arc<Mutex<SharedEkfState>>,
    shared_control: Arc<Mutex<Vector2<f64>>>,
    client_id: usize,
    mean_sender: mpsc::Sender<(f32, f32)>,
    cov_sender: mpsc::Sender<Matrix2<f32>>,
) {
    let peer_addr = stream.peer_addr().unwrap();
    println!("[Client {}] Connected from: {}", client_id, peer_addr);

    // Set non-blocking mode for fast reading
    // stream.set_nonblocking(true).unwrap();

    stream.set_nodelay(true).unwrap();

    let mut buffer = BufReader::new(&stream);

    println!("STREAM STARTED");

    loop {
        // Try to read from stream - non-blocking

        let mut buf = String::new();

        if let Err(e) = buffer.read_line(&mut buf) {
            println!("ERROR: {:?}", e);
            continue;
        }

        let buf = buf.trim().to_string();

        let data = match MeasurementData::parse(buf.as_str()) {
            Some(val) => val,
            None => continue,
        };

        if 340.0 * data.del_t > data.d {
            //            println!("[Client {}] Invalid measurement: skipping", client_id);
            continue;
        }

        let angle = wrap_to_pi((340.0 * data.del_t / data.d).asin());

        // println!(
        //     "[Client {}] {} , {} , {} , {} , {} , {}",
        //     client_id,
        //     data.timestamp,
        //     data.h,
        //     data.j,
        //     data.theta,
        //     data.d,
        //     angle.to_degrees()
        // );

        {
            let mut state = shared_state.lock().unwrap();

            // Check if this is a different client than the last update
            let should_update = match state.last_update_client {
                None => true,                                  // First update ever
                Some(last_client) => last_client != client_id, // Different client
            };

            if !should_update {
                // println!(
                //     "[Client {}] Same client as last update, skipping",
                //     client_id
                // );
                continue;
            }

            // Create measurement model for this observation
            let measurement_model = MeasurementModel::new(data.h, data.j, data.theta, 0.01);

            // Always predict before update
            let system_mode = SystemModel::new(0.01);

            {
                state
                    .ekf
                    .predict(&system_mode, &*shared_control.lock().unwrap());
            }
            // println!("[Client {}] Prediction step executed", client_id);

            // Update step with the angle measurement
            state.ekf.update(&measurement_model, angle);
            state.last_update_client = Some(client_id);
            state.update_count += 1;

            // Get updated state and covariance
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

            let covariance: Matrix2<f32> = Matrix2::from_vec(vec![
                covariance[(0, 0)] as f32,
                covariance[(0, 1)] as f32,
                covariance[(1, 0)] as f32,
                covariance[(1, 1)] as f32,
            ]);

            let _ = cov_sender.send(covariance);
            let _ = mean_sender.send((position[0] as f32, position[1] as f32));
        } // Mutex is unlocked here
    }
}

fn handle_controller_connection(
    mut stream: TcpStream,
    kf_state: Arc<Mutex<SharedEkfState>>,
    control: Arc<Mutex<Vector2<f64>>>,
) -> Result<(), ()> {
    stream.set_nodelay(true).unwrap();

    let mut buffer = BufReader::new(&stream);

    println!("STREAM STARTED");

    // Step 1: Parse the results and write it into

    let mut buf = String::new();

    if let Err(e) = buffer.read_line(&mut buf) {
        println!("ERROR: {:?}", e);
        return Err(());
    }

    let buf = buf.trim().to_string();

    let numbers: Vec<String> = buf.trim().split(',').map(|x| String::from(x)).collect();

    {
        let mut locked_control = match control.lock() {
            Ok(x) => *x,
            Err(_) => return Err(()),
        };

        locked_control[0] = numbers[0].parse().unwrap_or(0.0);
        locked_control[1] = numbers[1].parse().unwrap_or(0.0);
    }

    // Now the mutex is dropped, it is safe to lock the kalman filter state and
    // the current state to the controller
    {
        let locked_kalman = match kf_state.lock() {
            Ok(x) => x,
            Err(_) => return Err(()),
        };

        let state = locked_kalman.ekf.state;
        let response = format!("{},{},{}\n", state[0], state[1], state[2]);
        let _ = stream.write_all(response.as_bytes());
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let addr = "192.168.247.191:9099";

    // Create shared EKF state
    let shared_state = Arc::new(Mutex::new(SharedEkfState::new()));

    // no control initially
    let shared_control = Arc::new(Mutex::new(Vector2::<f64>::new(0.0, 0.0)));

    // Perform initial prediction
    {
        let mut state = shared_state.lock().unwrap();
        let system_model = SystemModel::new(0.01);
        {
            state
                .ekf
                .predict(&system_model, &*shared_control.lock().unwrap());
        }

        println!("EKF initialized at: {:?}", state.ekf.get_state());
    }

    // Bind TCP listener
    let listener = TcpListener::bind(addr)?;
    println!("Server listening on: {}", addr);

    let mut client_counter = 0;

    let (tx_mean, rx_mean) = std::sync::mpsc::channel::<(f32, f32)>();
    let (tx_cov, rx_cov) = std::sync::mpsc::channel::<Matrix2<f32>>();

    std::thread::spawn(move || {
        let rec = RecordingStreamBuilder::new("ekf_visualization")
            .spawn()
            .unwrap();

        let rerun_handler =
            RerunHandler::new(rec, String::from("ExtendedKalmanfilter"), rx_mean, rx_cov);

        rerun_handler.run();
    });

    // spawn a thread for listening for control signals

    // spawn a new thread for handling a single client

    let cloned_ss = Arc::clone(&shared_state);
    let cloned_ctrl = Arc::clone(&shared_control);

    thread::spawn(move || {
        let addr_input = "192.168.247.191:9100";

        let control_listener = TcpListener::bind(&addr_input).unwrap();
        println!("Server listening on: {}", &addr_input);

        for stream in control_listener.incoming() {
            let stream = match stream {
                Ok(val) => val,
                Err(_) => continue,
            };

            let _ = handle_controller_connection(stream, cloned_ss.clone(), cloned_ctrl.clone());
        }
    });

    // Accept connections in a loop
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let shared_state = Arc::clone(&shared_state);
                let shared_control = Arc::clone(&shared_control);
                client_counter += 1;
                let client_id = client_counter;

                let tx_mean = tx_mean.clone();
                let tx_cov = tx_cov.clone();

                // Spawn a new thread for each client
                thread::spawn(move || {
                    handle_client(
                        stream,
                        shared_state,
                        shared_control,
                        client_id,
                        tx_mean,
                        tx_cov,
                    );
                });
            }
            Err(e) => {
                eprintln!("Error accepting connection: {}", e);
            }
        }
    }

    Ok(())
}
