#include "saisieQT_main.h"

int helpMessage(const QApplication &app, QString text)
{
#ifdef WIN32
    QMessageBox msgBox(QMessageBox::NoIcon, app.applicationName(), text, QMessageBox::Ok);
    return msgBox.exec();
#else
    printf("\n%s\n", app.applicationName().toStdString().c_str());
    printf("\n%s", text.toStdString().c_str());
    return 0;
#endif
}

#if ( ( defined WIN32 ) && ( ELISE_QT_VERSION==5 ) )
class Win32CommandLineConverter
{
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


#if ( ( defined WIN32 ) && ( ELISE_QT_VERSION == 5 ) )
int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
{
    Win32CommandLineConverter cmd_line;
    int argc = cmd_line.argc();
    char **argv = cmd_line.argv();
#else
int main(int argc, char *argv[])
{
#endif

    QApplication app(argc, argv);

    app.setStyle("fusion");

    // QT Modifie le comportement de sscanf !!!!! problematique quand on parse les fichiers XML
    setlocale(LC_NUMERIC,"C");

    app.setOrganizationName("IGN");
    app.setApplicationName("QT graphical tools");

    QFile file(app.applicationDirPath() + "/../src/uti_qt/style.qss");
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        app.setStyleSheet(file.readAll());
        file.close();
    }

    // qt translations
    const QString locale = QLocale::system().name().section('_', 0, 0);
    QTranslator qtTranslator;
    qtTranslator.load(app.applicationName() + "_" + locale);
    app.installTranslator(&qtTranslator);

    QStringList cmdline_args = QCoreApplication::arguments();

    QString cmds = QObject::tr("Allowed commands:") + "\n\n" +
            QString("SaisieMasqQT\n") +
            QString("SaisieAppuisInitQT\n") +
            QString("SaisieAppuisPredicQT\n")+
            QString("SaisieBascQT\n")+
            QString("SaisieBoxQT\n\n");

    if (cmdline_args.size() > 1)
    {
        for (int i=0; i < cmdline_args.size(); ++i)
        {
            QString str = cmdline_args[i];
#ifdef _DEBUG
            cout << "\ncommand: " << str.toStdString().c_str()<<endl;
#endif

            if (!str.contains("SaisieQT"))
            {
                if (str.contains("SaisieMasqQT"))
                    saisieMasqQT_main(app, argc, argv);
                else if (str.contains("SaisieAppuisInitQT"))
                    saisieAppuisInitQT_main(app, argc, argv);
                else if (str.contains("SaisieAppuisPredicQT"))
                    saisieAppuisPredicQT_main(app, argc, argv);
                else if (str.contains("SaisieBoxQT"))
                    saisieBoxQT_main(app, argc, argv);
                else if (str.contains("SaisieBascQT"))
                    saisieBascQT_main(app, argc, argv);
                else
                {
                    QString text = str + QObject::tr(" is not a valid command!!!") + "\n\n" + cmds;
                    helpMessage(app, text);

                    return EXIT_FAILURE;
                }

                return EXIT_SUCCESS;
            }
        }
    }
    else
        helpMessage(app, cmds);

    return EXIT_SUCCESS;
}

bool checkNamePt(QString text)
{
    if (text.contains(".txt") && QFileInfo(text).isAbsolute())
    {
        QMessageBox::critical(NULL, "Error", "Don't use an absolute path for point file name!");
        return false;
    }
    return true;
}
