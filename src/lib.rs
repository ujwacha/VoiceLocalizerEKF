use glam::{f64, vec2};
use nalgebra::Matrix2;
use nalgebra::{SymmetricEigen, Vector2};
use rerun::Vec2D;
use rerun::{Color, Points2D, RecordingStream, external::glam};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub struct RerunHandler {
    pub rec: rerun::RecordingStream,
    pub name: String,
    pub mean_rx_mean: mpsc::Receiver<(f32, f32)>,
    pub mean_rx_cov: mpsc::Receiver<nalgebra::Matrix2<f32>>,
}

impl RerunHandler {
    pub fn new(
        rec: rerun::RecordingStream,
        name: String,
        mean_rx_mean: mpsc::Receiver<(f32, f32)>,
        mean_rx_cov: mpsc::Receiver<nalgebra::Matrix2<f32>>,
    ) -> Self {
        RerunHandler {
            rec,
            name,
            mean_rx_mean,
            mean_rx_cov,
        }
    }

    pub fn log_ekf_ellipse(
        &self,
        mean: Vector2<f32>,
        covariance_2d: Matrix2<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rec = &self.rec;
        let entity_path = &self.name;
        // Compute eigenvalues and eigenvectors
        let eigen = SymmetricEigen::new(covariance_2d);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Standard deviations along principal axes (1-sigma)
        let sigma1 = eigenvalues[0].abs().sqrt();
        let sigma2 = eigenvalues[1].abs().sqrt();

        // Generate ellipse points
        const NUM_POINTS: usize = 50;
        let mut positions = Vec::with_capacity(NUM_POINTS + 1);

        for i in 0..=NUM_POINTS {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / NUM_POINTS as f32;

            // Point on unit circle
            let circle_point = Vector2::new(angle.cos(), angle.sin());

            // Scale by standard deviations
            let scaled_point = Vector2::new(sigma1 * circle_point.x, sigma2 * circle_point.y);

            // Rotate by eigenvectors to align with covariance axes
            let rotated_point = eigenvectors * scaled_point;

            // Translate to mean position
            let final_point = mean + rotated_point;

            positions.push(glam::Vec3::new(final_point.x, final_point.y, 0.0));
        }

        // log the ellipse as a line strip (closed loop)
        rec.log(
            format!("{}/ellipse", entity_path),
            &rerun::LineStrips3D::new([positions])
                .with_radii([0.02])
                .with_colors([rerun::Color::from_unmultiplied_rgba(0, 200, 255, 200)]), // Cyan with transparency
        )?;

        // Log the mean position as a point
        rec.log(
            format!("{}/mean", entity_path),
            &rerun::Points3D::new([(mean.x, mean.y, 0.0)])
                .with_radii([0.05])
                .with_colors([rerun::Color::from_rgb(255, 0, 0)]), // Red point
        )?;

        // Optional: Log the principal axes as arrows
        let axis1 = eigenvectors.column(0) * sigma1;
        let axis2 = eigenvectors.column(1) * sigma2;

        rec.log(
            format!("{}/axis1", entity_path),
            &rerun::Arrows3D::from_vectors([glam::Vec3::new(axis1.x, axis1.y, 0.0)])
                .with_origins([glam::Vec3::new(mean.x, mean.y, 0.0)])
                .with_colors([rerun::Color::from_unmultiplied_rgba(255, 100, 100, 150)]),
        )?;

        rec.log(
            format!("{}/axis2", entity_path),
            &rerun::Arrows3D::from_vectors([glam::Vec3::new(axis2.x, axis2.y, 0.0)])
                .with_origins([glam::Vec3::new(mean.x, mean.y, 0.0)])
                .with_colors([rerun::Color::from_unmultiplied_rgba(100, 255, 100, 150)]),
        )?;

        rec.log(
            format!("{}/origin", entity_path),
            &Points2D::new(vec![vec2(0.0, 0.0)])
                .with_colors([rerun::Color::from_rgb(255, 255, 0)]) // Yellow
                .with_radii([0.05]),
        )?;

        Ok(())
    }

    pub fn run(&self) {
        loop {
            if let Ok(mean) = &self.mean_rx_mean.recv() {
                if let Ok(cov) = &self.mean_rx_cov.recv() {
                    let (x, y) = mean;

                    let mean = Vector2::from_vec(vec![*x, *y]);

                    &self.log_ekf_ellipse(mean, *cov);
                    let time = Duration::from_millis(16);
                    thread::sleep(time);
                }
            }
        }
    }
}
