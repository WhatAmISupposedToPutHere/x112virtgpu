use nix::libc;
use seccompiler::{
    BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
    SeccompRule,
};
use std::convert::TryInto;
use std::env::args;
use std::os::unix::process::CommandExt;
use std::process::{exit, Command};

fn main() {
    let args: Vec<_> = args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} [command]",
            args.get(0).unwrap_or(&("wrapper".into()))
        );
        exit(1);
    }

    let filter: BpfProgram = SeccompFilter::new(
        vec![(
            libc::SYS_memfd_create,
            vec![SeccompRule::new(vec![SeccompCondition::new(
                1,
                SeccompCmpArgLen::Dword,
                SeccompCmpOp::Eq,
                (libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) as u64,
            )
            .unwrap()])
            .unwrap()],
        )]
        .into_iter()
        .collect(),
        SeccompAction::Allow,
        SeccompAction::Errno(libc::ENOSYS.try_into().unwrap()),
        std::env::consts::ARCH.try_into().unwrap(),
    )
    .unwrap()
    .try_into()
    .unwrap();

    seccompiler::apply_filter(&filter).unwrap();

    Command::new(&args[1]).args(&args[2..]).exec();
}
