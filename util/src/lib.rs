use rand::Rng;
use std::fs::{remove_file, File};
use std::io::ErrorKind;

pub const SHM_PREFIX: &'static str = "/dev/shm/krshm-";
pub const SHM_DIR: &'static str = "/dev/shm/";

pub type Result<T> = std::result::Result<T, std::io::Error>;

pub fn create_shm_file(remove: bool) -> Result<([u8; 8], File)> {
    let mut rng = rand::thread_rng();
    loop {
        let shmem_name = [0x30u8; 8].map(|x| x + rng.gen_range(0..10));
        let mut shmem_path = SHM_PREFIX.to_owned();
        for c in shmem_name {
            shmem_path.push(c as char);
        }
        let result = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&shmem_path);
        match result {
            Ok(file) => {
                if remove {
                    remove_file(&shmem_path)?;
                }
                file.set_len(4)?;
                return Ok((shmem_name, file));
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            e => {
                e?;
            }
        }
    }
}
