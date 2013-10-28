#include <QtGui>
#include <QApplication>
#include "mainwindow.h"

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
    QApplication::setStyle("fusion");

#ifdef WIN32

	Win32CommandLineConverter cmd_line;

	int _argc = cmd_line.argc();

	QApplication app(_argc, cmd_line.argv());
#else
    QApplication app(argc, argv);
#endif
	
    app.setOrganizationName("IGN");
    app.setApplicationName("saisie");

    const QString locale = QLocale::system().name().section('_', 0, 0);

    // qt translations
    QTranslator qtTranslator;
    qtTranslator.load(app.applicationName() + "_" + locale);
    app.installTranslator(&qtTranslator);

    MainWindow w;

    QStringList cmdline_args = QCoreApplication::arguments();
    QString str;

    if (cmdline_args.size() > 1)
    {		
	
        for (int i=0; i< cmdline_args.size(); ++i)
        {
            bool removeArg = false;

            str = cmdline_args[i];

		if (str.contains("help"))
			{
				removeArg = true;				
#ifdef WIN32
				QMessageBox msgBox;
				msgBox.setText(	
								"Help command saisie\n"
								"\n"
								"saisie [filename] : open file (image or ply)\n"
								"\n"
								"Options\n"
								"\n"
								"Post	: Change postfix output file\n"
								"SzW	: Set size of windows\n"
								"Name	: Set file name output\n"
								"Gama	: Apply gamma on image\n"
								);
				msgBox.setStandardButtons(QMessageBox::Ok);
				w.close();
				return msgBox.exec();		
#else
				printf("Command saisie\n");
				printf("\n");
				printf("saisie [filename]   : open file (image or ply)\n");
				printf("Options :\n");
				printf("Post                : Change postfix of output file\n");
				printf("SzW                 : Set size of windows\n");
				printf("Name                : Set file name output\n");
				printf("Gama                : Apply gamma on image\n");				
				w.close();
				return 0;
#endif
			}
            if (str == "mode2D")
            {
                w.setMode2D(true);

                removeArg = true;
            }
            else
            {
				

                if (str.contains("Post="))
                {
                    w.setPostFix(str.mid(str.indexOf("Post=")+5, str.size()));

                    removeArg = true;
                }

                if (str.contains("SzW="))
                {
                    QString arg = str.mid(str.indexOf("SzW=")+4, str.size());
                    int szW = arg.toInt();

                    int szH = szW * w.height() / w.width();

                    w.resize( szW, szH );

                    removeArg = true;
                }

                if (str.contains("Name="))
                {
                    QString arg = str.mid(str.indexOf("Name=")+5, str.size());

                    w.getEngine()->setFilenameOut(arg);

                    removeArg = true;
                }

				if (str.contains("Gama="))
                {
                    QString strGamma = str.mid(str.indexOf("Gama=")+5, str.size());

					float aGamma = strGamma.toFloat();

                    w.setGamma(aGamma);

                    removeArg = true;
                }
            }

            if (removeArg)
            {
                cmdline_args[i] = cmdline_args.back();
                cmdline_args.pop_back();
                i--;
            }
        }
    }


    w.show();

    if (cmdline_args.size() > 1)
    {
        cmdline_args[0] = cmdline_args.back();
        cmdline_args.pop_back();

        w.addFiles(cmdline_args);
    }

    w.checkForLoadedData();

    return app.exec();
}

