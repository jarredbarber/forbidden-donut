extern crate crossterm;
extern crate nalgebra as na;
extern crate rand; // For dithering.

use crossterm::{cursor, QueueableCommand, Result};
use rand::Rng;
use std::cmp::{max, min};
use std::io::Write;

type Vec3 = na::Vector3<f32>;
type Point = na::Point3<f32>;
type Mat4 = na::Matrix4<f32>;

fn relu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

fn dither(i: f32, clip: usize) -> usize {
    let u = rand::thread_rng().gen::<f32>() - 0.5;
    let r = (i + u).round();
    if r < 0.0 {
        0
    } else {
        let r_i = r as usize;
        if r_i >= clip {
            clip - 1
        } else {
            r_i
        }
    }
}

fn main() -> Result<()> {
    let light_dir = Vec3::new(1.0, 5.0, -3.0).normalize();
    let cam_pos = Vec3::new(0.0, 0.0, 4.0);
    // Subdivisions of torus
    let (n1, n2) = (250, 100);
    // Radii of torus
    let (r1, r2) = (1.0, 0.45);
    let lightlevel_str = String::from("..',-~+*=$#@");

    let two_pi: f32 = 2.0 * 3.1415926535;
    let mut stdout = std::io::stdout();
    stdout.queue(cursor::Hide)?;

    let lightlevel: Vec<crossterm::style::Print<String>> = (0..lightlevel_str.len())
        .map(|i| crossterm::style::Print(lightlevel_str[i..(i + 1)].to_string()))
        .collect();

    let mut global_transform = Mat4::identity();
    loop {
        stdout.queue(crossterm::terminal::Clear(
            crossterm::terminal::ClearType::All,
        ))?;
        let (sx, sy) = crossterm::terminal::size().unwrap();
        let mut z_buffer = na::DMatrix::<f32>::repeat(sx as usize, sy as usize, 1000.0);

        let z_clip = 1000.0;
        let aspect = (min(sx, sy) as f32) / (max(sx, sy) as f32);
        let screenspace = Mat4::new_translation(&Vec3::new(0.5 * sx as f32, 0.5 * sy as f32, 0.0))
            * Mat4::new_scaling(0.5 * min(sx, sy) as f32)
            * Mat4::new_perspective(aspect, 3.141 / 4.0, 0.1, z_clip)
            * Mat4::new_translation(&cam_pos);
        z_buffer.fill(-z_clip);

        // For each voxel, compute screenspace position, lighting, then (maybe) draw.
        for i1 in 0..n1 {
            let phi1 = two_pi * (i1 as f32) / (n1 as f32);
            let rot: Mat4 = Mat4::from_euler_angles(0.0, 0.0, phi1);

            for i2 in 0..n2 {
                // Compute screenspace position + worldspace normal (for lighting)
                let (p_world, p_screen, n) = {
                    let phi2 = two_pi * (i2 as f32) / n2 as f32;
                    // cp = circle point; cn = circle normal.
                    let cp = Point::new(r2 * phi2.cos() + r1, 0.0, r2 * phi2.sin());
                    let cn = Vec3::new(phi2.cos(), 0.0, phi2.sin());

                    // To object space (isometry)
                    let p1 = rot.transform_point(&cp);
                    let n1 = rot.transform_vector(&cn);

                    // To world space (isometry)
                    let p2 = global_transform.transform_point(&p1);
                    let n2 = global_transform.transform_vector(&n1);

                    // p3 goes to screen space (homogenous)
                    let p3 = screenspace.transform_point(&p2);
                    // Technically, n2 should still be normalized
                    (p2, p3, n2.normalize())
                };

                // Unit vector pointing from p_world to the camera
                let cam_vec = (cam_pos - (p_world - Point::origin())).normalize();

                if !(p_screen.x < 0.0
                    || p_screen.y < 0.0
                    || cam_vec.dot(&n) > 0.0
                    || p_screen.x >= sx as f32
                    || p_screen.y >= sy as f32)
                {
                    let light = {
                        // Phong shading model
                        let a = relu(n.dot(&light_dir));
                        let r = 2.0 * a * n.dot(&cam_vec) - light_dir.dot(&cam_vec);
                        let light = 0.75 * a + 0.25 * r * r * r;
                        if light > 0.99 {
                            0.99
                        } else {
                            light
                        }
                    };
                    if light > 0.0 {
                        let (ix, iy) = (
                            dither(p_screen.x, sx as usize),
                            dither(p_screen.y, sy as usize),
                        );
                        let this_z = z_buffer.get_mut((ix, iy)).unwrap();
                        if p_screen.z > *this_z {
                            *this_z = p_screen.z;
                            let light_dithered =
                                dither(light * lightlevel.len() as f32, lightlevel.len());
                            stdout.queue(cursor::MoveTo(ix as u16, iy as u16))?;
                            stdout.queue(&lightlevel[light_dithered as usize])?;
                        }
                    }
                }
            }
        }

        global_transform *= Mat4::from_euler_angles(0.0, 0.0, 0.03);
        global_transform *= Mat4::from_euler_angles(0.1, -0.05, 0.0);

        stdout.queue(cursor::MoveTo(sx / 2 - 14, 1))?;
        stdout.queue(crossterm::style::Print("F O R B I D D E N D O N U T"))?;
        stdout.queue(cursor::MoveTo(sx / 2 - 14, sy - 1))?;
        stdout.queue(crossterm::style::Print("F O R B I D D E N D O N U T"))?;

        stdout.flush()?;
        std::thread::sleep_ms(80);
    }
}
