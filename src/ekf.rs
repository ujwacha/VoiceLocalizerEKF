use nalgebra::{Matrix2, SMatrix, Vector2};

/// Extended Kalman Filter for 2D state estimation
pub struct ExtendedKalmanFilter {
    /// State vector [x, y]
    pub state: Vector2<f64>,
    /// Covariance matrix
    pub covariance: Matrix2<f64>,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF with initial state and covariance
    pub fn new(initial_state: Vector2<f64>, initial_covariance: Matrix2<f64>) -> Self {
        Self {
            state: initial_state,
            covariance: initial_covariance,
        }
    }

    /// Prediction step
    pub fn predict(&mut self, system_model: &SystemModel) {
        // State prediction: x_pred = f(x)
        self.state = system_model.predict_state(&self.state);

        // Covariance prediction: P_pred = F * P * F^T + Q
        let f = system_model.jacobian(&self.state);
        self.covariance = f * self.covariance * f.transpose() + system_model.process_noise;
    }

    /// Update step
    pub fn update(&mut self, measurement_model: &MeasurementModel, measurement: f64) {
        // Innovation (measurement residual)
        let predicted_measurement = measurement_model.predict_measurement(&self.state);
        let innovation = measurement - predicted_measurement;

        // Measurement Jacobian
        let h = measurement_model.jacobian(&self.state);

        // Innovation covariance: S = H * P * H^T + R
        let s = h * self.covariance * h.transpose() + measurement_model.measurement_noise;

        // Kalman gain: K = P * H^T * S^-1
        // For 1x1 matrix, inversion is just 1/value
        let s_inv = 1.0 / s[(0, 0)];
        let kalman_gain = self.covariance * h.transpose() * s_inv;

        // State update: x = x + K * innovation
        self.state = self.state + kalman_gain * innovation;

        // Covariance update: P = (I - K * H) * P
        let i_kh = Matrix2::identity() - kalman_gain * h;
        self.covariance = i_kh * self.covariance;
    }

    /// Get current state estimate
    pub fn get_state(&self) -> Vector2<f64> {
        self.state
    }

    /// Get current covariance
    pub fn get_covariance(&self) -> Matrix2<f64> {
        self.covariance
    }
}

/// System model for the robot (constant position model)
pub struct SystemModel {
    /// Process noise covariance
    pub process_noise: Matrix2<f64>,
}

impl SystemModel {
    pub fn new(process_noise_variance: f64) -> Self {
        Self {
            process_noise: Matrix2::identity() * process_noise_variance,
        }
    }

    /// State transition function: f(x) = x (constant position)
    pub fn predict_state(&self, state: &Vector2<f64>) -> Vector2<f64> {
        *state
    }

    /// Jacobian of state transition function (identity for constant model)
    pub fn jacobian(&self, _state: &Vector2<f64>) -> Matrix2<f64> {
        Matrix2::identity()
    }
}

/// Measurement model for angle measurements
pub struct MeasurementModel {
    /// Sensor position (h, k)
    pub h_pos: f64,
    pub k_pos: f64,
    /// Sensor orientation (theta)
    pub theta: f64,
    /// Measurement noise variance (as 1x1 matrix)
    pub measurement_noise: SMatrix<f64, 1, 1>,
}

impl MeasurementModel {
    pub fn new(h: f64, k: f64, theta: f64, measurement_noise: f64) -> Self {
        Self {
            h_pos: h,
            k_pos: k,
            theta,
            measurement_noise: SMatrix::<f64, 1, 1>::new(measurement_noise),
        }
    }

    /// Measurement function: h(x) = atan2(y - k, x - h) - theta
    pub fn predict_measurement(&self, state: &Vector2<f64>) -> f64 {
        let vec_i = state[0] - self.h_pos;
        let vec_j = state[1] - self.k_pos;
        vec_j.atan2(vec_i) - self.theta
    }

    /// Jacobian of measurement function
    /// H = [-vec_j / magn, vec_i / magn]
    /// where magn = vec_i^2 + vec_j^2
    pub fn jacobian(&self, state: &Vector2<f64>) -> SMatrix<f64, 1, 2> {
        let vec_i = state[0] - self.h_pos;
        let vec_j = state[1] - self.k_pos;
        let magn = vec_i * vec_i + vec_j * vec_j;

        SMatrix::<f64, 1, 2>::from_row_slice(&[-vec_j / magn, vec_i / magn])
    }
}

