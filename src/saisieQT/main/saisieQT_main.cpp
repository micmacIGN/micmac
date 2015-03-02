#include "saisieQT_main.h"

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

#if ( ( defined WIN32 ) && ( ELISE_QT_VERSION==5 || ELISE_QT_VERSION==4 ) )
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


#if ( ( defined WIN32 ) && ( ELISE_QT_VERSION == 5 || ELISE_QT_VERSION==4) )
int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
{
	Win32CommandLineConverter cmd_line;
	int argc = cmd_line.argc();
	char **argv = cmd_line.argv();
#else
int main(int argc, char *argv[])
{
#endif

	MMD_InitArgcArgv( argc, argv );
	initQtLibraryPath();

	QApplication app(argc, argv);

	//app.setStyle("fusion");

	// QT Modifie le comportement de sscanf !!!!! problematique quand on parse les fichiers XML
	setlocale(LC_NUMERIC,"C");

	app.setOrganizationName("Culture3D");
	app.setApplicationName("QT graphical tools");

	setStyleSheet(app);

	//qDebug() << "Number of screens:" << QGuiApplication::screens().size();

	//QScreen *scrre =  QGuiApplication::primaryScreen();

	//qDebug() << scrre->size();

	QString cmds = QObject::tr("Allowed commands:") + "\n\n" +
		QString("SaisieMasqQT\n") +
		QString("SaisieAppuisInitQT\n") +
		QString("SaisieAppuisPredicQT\n")+
		QString("SaisieBascQT\n")+
		QString("SaisieBoxQT\n\n");

	if (argc > 1)
	{
		for (int i=0; i < argc; ++i)
		{
			QString str(argv[i]);

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
