mod ekf;
use ekf::{ExtendedKalmanFilter, MeasurementModel, SystemModel};
use ekf_server::RerunHandler;
use nalgebra::{Matrix2, Vector2};
use rerun::{RecordingStream, RecordingStreamBuilder};
use std::io::{BufRead, BufReader, Read};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

/// Shared EKF state across all connections
struct SharedEkfState {
    ekf: ExtendedKalmanFilter,
    last_update_client: Option<usize>,
    update_count: usize,
}

impl SharedEkfState {
    fn new() -> Self {
        // Initialize state at (0.247, 0.635)
        let initial_state = Vector2::new(0.247, 0.635);
        let initial_covariance = Matrix2::identity() * 0.01;
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

        let angle = (340.0 * data.del_t / data.d).asin();

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
            state.ekf.predict(&system_mode);
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

fn main() -> std::io::Result<()> {
    let addr = "192.168.9.172:9099";

    // Create shared EKF state
    let shared_state = Arc::new(Mutex::new(SharedEkfState::new()));

    // Perform initial prediction
    {
        let mut state = shared_state.lock().unwrap();
        let system_model = SystemModel::new(0.01);
        state.ekf.predict(&system_model);
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

    // Accept connections in a loop
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let shared_state = Arc::clone(&shared_state);
                client_counter += 1;
                let client_id = client_counter;

                let tx_mean = tx_mean.clone();
                let tx_cov = tx_cov.clone();

                // Spawn a new thread for each client
                thread::spawn(move || {
                    handle_client(stream, shared_state, client_id, tx_mean, tx_cov);
                });
            }
            Err(e) => {
                eprintln!("Error accepting connection: {}", e);
            }
        }
    }

    Ok(())
}
