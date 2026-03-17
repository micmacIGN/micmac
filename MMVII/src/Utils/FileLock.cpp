#include "FileLock.h"
#include "MMVII_DeclareCste.h"
#include "cMMVII_Appli.h"

#ifdef WIN32
#  include <windows.h>
#  include <filesystem>
#  include <sstream>
#  include <iomanip>
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <filesystem>
#  include <unordered_set>
#  include <stdexcept>
#endif

namespace MMVII {

#ifdef WIN32

FileLock::FileLock(const std::string& filePath)
{
    auto mutexName = std::filesystem::weakly_canonical(filePath).wstring();
    if (mutexName.length() < MAX_PATH) {
        std::replace(mutexName.begin(), mutexName.end(), L'\\', L'/');
    } else {
        auto key1 = std::hash<std::wstring>{}(mutexName);
        auto key2 = std::hash<std::wstring>{}(std::wstring(mutexName.rbegin(), mutexName.rend()));
        std::wstringstream ss;
        ss << std::hex << std::setw(16) << std::setfill(L'0') << key1 << L"/" << key2;
        mutexName = ss.str();
    }
    mHandle = CreateMutexW(NULL, FALSE, mutexName.c_str());
    if (!mHandle) {
        MMVII_INTERNAL_ERROR("Can't create Mutex on file '" + filePath + "'");
    }
    DWORD wait = WaitForSingleObject(mHandle, INFINITE);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) {
        MMVII_INTERNAL_ERROR("Can't wait for Mutex on file '" + filePath + "'");
    }
}

FileLock::~FileLock()
{
    ReleaseMutex(mHandle);
    CloseHandle(mHandle);
    mHandle = nullptr;
}

#else // !WIN32

static std::string get_lock_dir()
{
    static std::unordered_set<std::string> created_dirs;

    std::string dir = cMMVII_Appli::CurrentAppli().DirProject()
                      + MMVII::TmpMMVIIDirGlob
                      + "/Locks/";

    if (created_dirs.find(dir) == created_dirs.end()) {
        std::filesystem::create_directories(dir);
        created_dirs.insert(dir);
    }
    return dir;
}

static std::string make_lock_path(const std::string& filepath)
{
    std::string abs  = std::filesystem::absolute(filepath).string();
    size_t      h    = std::hash<std::string>{}(abs);
    std::string name = std::filesystem::path(abs).filename().string();
    return get_lock_dir() + name + "_" + std::to_string(h) + ".lock";
}

FileLock::FileLock(const std::string& filepath)
{
    const std::string lock_path = make_lock_path(filepath);

    lock_fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd_ < 0)
    {
        MMVII_INTERNAL_ERROR("Can't open for locking: '" + filepath + "'");
    }

    struct flock fl{};
    fl.l_type   = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start  = 0;
    fl.l_len    = 0;

    if (fcntl(lock_fd_, F_SETLKW, &fl) < 0)
    {
        close(lock_fd_);
        MMVII_INTERNAL_ERROR("Can't lock file '" + filepath + "'");
    }
}

FileLock::~FileLock()
{
    struct flock fl{};
    fl.l_type   = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start  = 0;
    fl.l_len    = 0;

    fcntl(lock_fd_, F_SETLK, &fl);
    close(lock_fd_);
}

#endif // !WIN32

} // namespace MMVII
