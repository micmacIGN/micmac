#ifndef FILELOCK_H
#define FILELOCK_H

#include <string>

namespace MMVII {

/// @brief RAII file lock providing mutual exclusion between multiple processes
///        accessing the same file.
///
/// On Linux, FileLock creates a companion lock file in a dedicated directory:
///   cMMVII_Appli::CurrentAppli().DirProject() + MMVII::TmpMMVIIDirGlob + "/Locks/"
/// and acquires a POSIX advisory write lock (fcntl F_SETLKW) on it.
/// The lock directory is created on first use for each unique path.
///
/// On Windows, FileLock uses a named Mutex derived from the canonical path
/// of the file to protect. If the path exceeds MAX_PATH, a hash-based name
/// is used instead.
///
/// @note Locks are advisory: all cooperating processes must use FileLock consistently.
/// @note On Linux, fcntl locks are per (pid, inode): threads within the same process
///       do NOT block each other. Add a std::mutex if intra-process exclusion is needed.
/// @note The lock is automatically released if the process crashes (kernel/OS cleanup).
/// @note FileLock is non-copyable and non-movable.
///
/// @par Example
/// @code
/// {
///     FileLock lock("data.txt");
///     black_box_write("data.txt");
/// } // lock released here
/// @endcode
class FileLock {
public:
    /// @brief Acquires an exclusive lock on the given file.
    ///
    /// On Linux, opens (or creates) a companion lock file and blocks until
    /// the POSIX write lock is obtained. The lock files are created in
    /// {DirProject}/MMVII-Tmp-Dir-Glob/Locks/
    ///
    /// On Windows, creates or opens a named Mutex derived from the file's
    /// canonical path and waits indefinitely until it is acquired.
    ///
    /// @param filepath Path to the file to protect. The file does not need
    ///                 to exist at the time the lock is acquired.
    /// @throws MMVII_INTERNAL_ERROR if the lock file cannot be opened
    ///         or the lock cannot be acquired.
    explicit FileLock(const std::string& filepath);

    /// @brief Releases the lock.
    ///
    /// On Linux, unlocks the POSIX lock and closes the lock file descriptor.
    /// The companion lock file itself is NOT deleted on destruction.
    /// On Windows, releases and closes the Mutex handle.
    ~FileLock();

    FileLock(const FileLock&)            = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
#ifdef WIN32
    void* mHandle = nullptr; ///< Windows Mutex handle.
#else
    int lock_fd_; ///< File descriptor of the companion lock file (Linux).
#endif
};

} // namespace MMVII

#endif // FILELOCK_H
