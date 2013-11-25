#include "saisieMasqQT_main.h"

bool MMVisualMode = false;

#ifdef _WIN32
class Win32CommandLineConverter;

class Win32CommandLineConverter {
private:
	std::unique_ptr<char*[]> argv_;
	std::vector<std::unique_ptr<char[]>> storage_;
public:
	Win32CommandLineConverter()
	{
		LPWSTR cmd_line = GetCommandLineW();
		int argc;
		LPWSTR* w_argv = CommandLineToArgvW(cmd_line, &argc);
		argv_ = std::unique_ptr<char*[]>(new char*[argc]);
		storage_.reserve(argc);
		for(int i=0; i<argc; ++i) {
			storage_.push_back(ConvertWArg(w_argv[i]));
			argv_[i] = storage_.back().get();
		}
		LocalFree(w_argv);
	}
	int argc() const
	{
		return static_cast<int>(storage_.size());
	}
	char** argv() const
	{
		return argv_.get();
	}
	static std::unique_ptr<char[]> ConvertWArg(LPWSTR w_arg)
	{
		int size = WideCharToMultiByte(CP_UTF8, 0, w_arg, -1, nullptr, 0, nullptr, nullptr);
		std::unique_ptr<char[]> ret(new char[size]);
		WideCharToMultiByte(CP_UTF8, 0, w_arg, -1, ret.get(), size, nullptr, nullptr);
		return ret;
	}
};
#endif



#ifdef WIN32
int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
#else
int main(int argc, char *argv[])
#endif
{
    cout << "here" <<endl;
    QApplication::setStyle("fusion");

#ifdef WIN32

    Win32CommandLineConverter cmd_line;

    int _argc = cmd_line.argc();

    QApplication app(_argc, cmd_line.argv());
#else
    QApplication app(argc, argv);
#endif

    app.setOrganizationName("IGN");
    app.setApplicationName("SaisieQT");

    QStringList cmdline_args = QCoreApplication::arguments();
    QString str;

    if (cmdline_args.size() > 1)
    {
        for (int i=0; i< cmdline_args.size(); ++i)
        {
            bool removeArg = false;

            str = cmdline_args[i];

            if (str.contains("SaisieMasqQT"))
            {
                saisieMasqQT_main(app);
                removeArg = true;
            }
            else if (str.contains("SaisieAppuisInitQT"))
            {
                //saisieAppuisInitQT_main(app);
                removeArg =true;
            }

            if (removeArg)
            {
                cmdline_args[i] = cmdline_args.back();
                cmdline_args.pop_back();
                i--;
            }
        }
    }


}


