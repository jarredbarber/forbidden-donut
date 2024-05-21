use crossterm::{cursor, QueueableCommand};
use rand::Rng;
use std::cmp::{max, min};
use std::io::Write;

type Vec3 = nalgebra::Vector3<f32>;
type Point = nalgebra::Point3<f32>;
type Mat4 = nalgebra::Matrix4<f32>;
type Result<T> = std::result::Result<T, std::io::Error>;

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

struct FrameBuffer {
    brightness: Vec<u8>,
    z_buffer: Vec<f32>,
    sx: usize,
    sy: usize,
}

impl FrameBuffer {
    fn new() -> Result<FrameBuffer> {
        let (sx_, sy_) = crossterm::terminal::size().unwrap();
        let sx = sx_ as usize;
        let sy = sy_ as usize;
        let size = ((sx + 1) * sy) as usize;

        std::io::stdout().queue(cursor::Hide)?;

        let brightness: Vec<u8> = Vec::with_capacity(size);
        let z_buffer: Vec<f32> = Vec::with_capacity(size);
        Ok(FrameBuffer {
            sx,
            sy,
            brightness,
            z_buffer,
        })
    }

    fn clear(&mut self) {
        let (sx, sy) = crossterm::terminal::size().unwrap();
        self.sx = sx as usize;
        self.sy = sy as usize;
        let size = self.sy * (self.sx + 1);
        self.z_buffer.clear();
        self.z_buffer.resize(size, -1000.0);
        self.brightness.clear();
        self.brightness.resize(size, ' ' as u8);
        for y in 0..self.sy {
            self.brightness[y * (self.sx + 1) + self.sx] = '\n' as u8;
        }
    }

    fn write(&self) -> Result<()> {
        let mut stdout = std::io::stdout();
        stdout.queue(crossterm::terminal::Clear(
            crossterm::terminal::ClearType::All,
        ))?;
        stdout.queue(cursor::MoveTo(0, 0))?;
        // actually safe
        let s = unsafe { std::str::from_utf8_unchecked(&self.brightness[0..(self.sx * self.sy)]) };
        stdout.queue(crossterm::style::Print(&s))?;
        Ok(())
    }

    fn poke_if(&mut self, x: usize, y: usize, value: f32, z: f32) {
        let lightlevel_str = "-~+*=;%#$@";
        let n = lightlevel_str.len();

        let ix = y * (self.sx + 1) + x;

        if self.z_buffer[ix] < z {
            self.z_buffer[ix] = z;
            let val_ix = dither(value * (n as f32), n);
            self.brightness[ix] = lightlevel_str.as_bytes()[val_ix];
        }
    }
}

fn main() -> Result<()> {
    let light_dir = Vec3::new(1.0, 5.0, -3.0).normalize();
    let cam_pos = Vec3::new(0.0, 0.0, 4.0);
    // Subdivisions of torus
    let (n1, n2) = (500, 200);
    // Radii of torus
    let (r1, r2) = (1.0, 0.45);

    let two_pi: f32 = 2.0 * 3.1415926535;
    let mut stdout = std::io::stdout();
    stdout.queue(cursor::Hide)?;

    let mut global_transform = Mat4::identity();

    let mut framebuffer = FrameBuffer::new()?;
    loop {
        framebuffer.clear();
        let (sx, sy) = (framebuffer.sx, framebuffer.sy);

        let aspect = (min(sx, sy) as f32) / (max(sx, sy) as f32);
        let screenspace = Mat4::new_translation(&Vec3::new(0.5 * sx as f32, 0.5 * sy as f32, 0.0))
            * Mat4::new_scaling(0.5 * min(sx, sy) as f32)
            * Mat4::new_perspective(aspect, 3.141 / 4.0, 0.1, 1000.0)
            * Mat4::new_translation(&cam_pos);

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
                        framebuffer.poke_if(ix, iy, light, p_screen.z);
                    }
                }
            }
        }

        global_transform *= Mat4::from_euler_angles(0.0, 0.0, 0.03);
        global_transform *= Mat4::from_euler_angles(0.1, -0.05, 0.0);

        framebuffer.write()?;
        stdout.queue(cursor::MoveTo(sx as u16 / 2 - 14, 1))?;
        stdout.queue(crossterm::style::Print("F O R B I D D E N D O N U T"))?;
        stdout.queue(cursor::MoveTo(sx as u16 / 2 - 14, sy as u16 - 1))?;
        stdout.queue(crossterm::style::Print("F O R B I D D E N D O N U T"))?;

        stdout.flush()?;
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
