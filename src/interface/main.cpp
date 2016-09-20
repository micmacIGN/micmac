/*
Ce programme est une interface graphique pour un ensemble d'outils de traitement de chantiers photogrammétrique dont Pastis, Apero et MICMAC.
Pour exécuter ce programme, il faut respecter l'arborescence :
- micmac/bin/ contenant les binaires (outils de traitement de chantier)
- micmac/bin/include/XML_GEN/ParamChantierPhotogram.xml
- dans le dossier de l'interface : interfaceMicmac (l'exécutable), english.qm (la traduction compilée en Anglais)
- dossier de l'interface/help/ contenant assistant (outil d'aide), help.qch, help.qhc (fichiers d'aide compilés)
- dossier de l'interface/lib/ contenant tiff2rgba (outil de conversion de format d'image), liQtCore.so, libQtGui.so, libQtXml.so, libQtOpenGL.so, libGL.so, libGLU.so, libpthread.so (librairies Qt et openGL)
- dossier de l'interface/xml/BDCamera.xml (sauvegarde des paramètres des caméras utilisées)


*/

#include "all.h"



int main(int argc, char *argv[])
{	
	MMD_InitArgcArgv(argc, argv);

	g_interfaceDirectory = QString( ( NS_ParamChantierPhotogram::MMDir()+"/interface/" ).c_str() );
	g_iconDirectory = g_interfaceDirectory+"images/";

	QApplication app(argc, argv);
	
	if (argc>2) {
		cout << "interfaceMicmac [sauvegarde.xml]" << endl;
		return -1;
	}
	
	//fermeture si une interface est déjà lancée ailleurs
	/*QSharedMemory sharedMemory("{35DF4G354DFH35FG5-534GN53-21S}");
	if (sharedMemory.create(sizeof(int))==false) {
		int reponse = Interface::dispMsgBox("L'application est déjà lancée dans une autre fenêtre.", "Voulez-vous l'exécuter quand même ?", QVector<int>()<<0<<-1<<1, 2);
		if (reponse!=0) return 0;
	}*/	//pb : le segment reste si l'interface est quittée brusquement
	
	//rappel des données globales inter-sessions
	QSettings settings("IGN/MATIS", "interfaceMicmac");
	QCoreApplication::setOrganizationName("IGN/MATIS");
	QCoreApplication::setOrganizationDomain("IGN/MATIS/interfaceMicmac.com");
	QCoreApplication::setApplicationName("interfaceMicmac");
	
	//traductions (doivent être faites avant de charger la MainWindow) et encodage
	QString langue = settings.value("langue").toString();

	QTranslator qtTranslator;
	if ( !qtTranslator.load( g_interfaceDirectory+"french.qm" ) )
		cout << "unable to load french dictionnary" << endl;
	if ((langue.isEmpty() && QLocale::system().language()==QLocale::English) || (langue==QLocale::languageToString(QLocale::English))) {
		if (langue.isEmpty()) settings.setValue("langue", QLocale::languageToString(QLocale::English));
		QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
		QTextCodec::setCodecForTr(QTextCodec::codecForName("UTF-8"));
		QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
	} else {
		if (langue.isEmpty()) settings.setValue("langue", QLocale::languageToString(QLocale::French));
		app.installTranslator(&qtTranslator);
		QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
		QTextCodec::setCodecForTr(QTextCodec::codecForName("UTF-8"));
		QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
	}
	
	//style
	app.setFont(QFont("Arial", 12));

	//interface
	Interface *interf = new Interface(settings);
	interf->show();
	if (argc==2) interf->openCalc(QString(argv[1]));
	
	return app.exec();
}


/*

exécutables annexes (à modifier pour windows et mac) :
kill all
make
mv (dans make et avec *)
cd
sh (utilisé par les binaires)
cp (utilisé par les binaires)

*/
