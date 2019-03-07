#include "saisieQT_main.h"

const cArgLogCom cArgLogCom::NoLog(-1);

vector<cMMCom> getSaisieQtCommands()
{
	vector<cMMCom> result;

	result.push_back(cMMCom("SaisieAppuisInitQT", saisieAppuisInitQT_main, " Interactive tool for initial capture of GCP"));
	result.push_back(cMMCom("SaisieAppuisPredicQT", saisieAppuisPredicQT_main, " Interactive tool for assisted capture of GCP"));
	result.push_back(cMMCom("SaisieBascQT", saisieBascQT_main, " Interactive tool to capture information on the scene"));
	result.push_back(cMMCom("SaisieCylQT", SaisieCylQT_main, " Interactive tool to capture information on the scene for cylinders"));
	result.push_back(cMMCom("SaisieMasqQT", saisieMasqQT_main, " Interactive tool to capture masq"));
	result.push_back(cMMCom("SaisieBoxQT", saisieBoxQT_main, " Interactive tool to capture 2D box"));

	return result;
}

int qtpopup(const string &aText)
{
	return QMessageBox(QMessageBox::NoIcon, QString("popup"), QString(aText.c_str()), QMessageBox::Ok).exec();
}

int helpMessage(const QApplication &app, QString text)
{
    QString title = QObject::tr("Help for ") + app.applicationName();
#ifdef WIN32
    QMessageBox msgBox(QMessageBox::NoIcon, title, text, QMessageBox::Ok);
    return msgBox.exec();
#else
    printf("\n%s\n", title.toStdString().c_str());
    printf("\n%s", text.toStdString().c_str());
    return 0;
#endif
}

#if ( ( defined WIN32 ) && ( ELISE_QT ) )
class Win32CommandLineConverter
{
private:
    char** argv_;
    std::vector<char*> storage_;
public:
    Win32CommandLineConverter()
    {
        LPWSTR cmd_line = GetCommandLineW();

        int argc;

        LPWSTR* w_argv = CommandLineToArgvW(cmd_line, &argc);

        argv_ = new char*[argc];

        storage_.reserve(argc);

        for(int i=0; i<argc; ++i) {
            storage_.push_back(ConvertWArg(w_argv[i]));
            argv_[i] = storage_.back();
        }
        LocalFree(w_argv);
    }
    int argc() const
    {
        return static_cast<int>(storage_.size());
    }
    char** argv() const
    {
        return argv_;
    }
    static char* ConvertWArg(LPWSTR w_arg)
    {
        int size = WideCharToMultiByte(CP_UTF8, 0, w_arg, -1, nullptr, 0, nullptr, nullptr);
        char* ret = new char[size];
        WideCharToMultiByte(CP_UTF8, 0, w_arg, -1, ret, size, nullptr, nullptr);
        return ret;
    }
};
#endif

QApplication & getQApplication()
{
	QApplication *result = (QApplication *)QCoreApplication::instance();
	if (result == NULL) ELISE_ERROR_EXIT("no QApplication available");
	return *result;
}

//~ #if ( ( defined WIN32 ) && ( ELISE_QT ) )
//~ int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
//~ {
    //~ Win32CommandLineConverter cmd_line;
    //~ int argc = cmd_line.argc();
    //~ char **argv = cmd_line.argv();
//~ #else
int main(int argc, char *argv[])
{
	//~ gDefaultDebugErrorHandler->setAction(MessageHandler::CIN_GET);
//~ #endif

    MMD_InitArgcArgv( argc, argv );
    initQtLibraryPath();

    QApplication app(argc, argv);

    // QT Modifie le comportement de sscanf !!!!! problematique quand on parse les fichiers XML
    setlocale(LC_NUMERIC,"C");

    app.setOrganizationName("Culture3D");
    app.setApplicationName("QT graphical tools");

    setStyleSheet(app);

	vector<cMMCom> commands = getSaisieQtCommands();

	tCommande commandFunction = NULL;
	string commandStr, lowCommandStr;
	if (argc > 1)
	{
		commandStr = argv[1];
		lowCommandStr = StrToLower(commandStr);
	}

	for (size_t iCommand = 0; iCommand < commands.size(); iCommand++)
		if (lowCommandStr == commands[iCommand].mLowName) commandFunction = commands[iCommand].mCommand;

	if (commandFunction == NULL)
	{
		ostringstream ss;

		if ( !commandStr.empty())
			ss << "[" << commandStr << "]" << QObject::tr(" is not a valid command!!!").toStdString() << "\n" << endl;

		ss << "valid commands are: \n" << endl;
		for (size_t iCommand = 0; iCommand < commands.size(); iCommand++)
			ss << commands[iCommand].mName << endl;
		ss << endl;

		helpMessage(app, QString(ss.str().c_str()));

		return EXIT_FAILURE;
	}

	return (*commandFunction)(argc, argv);
}

bool checkNamePt(QString text)
{
    if (text.contains(".txt") && QFileInfo(text).isAbsolute())
    {
        QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("Don't use an absolute path for point file name!"));
        return false;
    }
    return true;
}

QStringList getFilenames(string aDir, string aName)
{
    list<string> aNamelist = RegexListFileMatch(aDir, aName, 1, false);
    QStringList filenames;

    for
        (
        list<string>::iterator itS=aNamelist.begin();
    itS!=aNamelist.end();
    itS++
        )
        filenames.push_back( QString((aDir + *itS).c_str()));

    return filenames;
}

void loadTranslation(QApplication &app)
{
    QSettings settings(QApplication::organizationName(), QApplication::applicationName());
    settings.beginGroup("Language");
    int lang = settings.value("lang", 0).toInt();
    settings.endGroup();

    if(lang>0)
    {
        QString sLang = "saisie_";
        QString path = app.applicationDirPath() + QDir::separator() + "../include/qt/translations";

        //cf Settings.h
        if (lang == 1)
            sLang += "fr";
        else if (lang == 2)
            sLang += "es";
        else if (lang == 3)
            sLang += "cn";
        else if (lang == 4)
            sLang += "ar";
        else if (lang == 5)
            sLang += "ru";

        sLang += ".qm";

        QTranslator *qtTranslator = new QTranslator(QApplication::instance());

        if ( qtTranslator->load(sLang, path) )
        {
            QApplication::instance()->installTranslator(qtTranslator);
        }
        else
        {
            QMessageBox::critical(NULL, "Error", "Can't load translation file: " + sLang + "\n" +
                "In: " + path);
        }
    }
}

void updateSettings(QSettings &settings, Pt2di aSzWin, Pt2di aNbFen, bool aForceGray)
{
    settings.beginGroup("MainWindow");
    if (aSzWin.x > 0)
        settings.setValue("size", QSize(aSzWin.x, aSzWin.y));
    else if (!settings.contains("MainWindow/size"))
    {
        settings.setValue("size", QSize(800, 800));
        aSzWin.x = aSzWin.y = 800;
    }
    settings.setValue("NbFen", QPoint(aNbFen.x, aNbFen.y));
    settings.endGroup();

    settings.beginGroup("Drawing settings");
    settings.setValue("forceGray",     aForceGray  );
    settings.endGroup();
}
