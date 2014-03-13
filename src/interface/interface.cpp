#include  "interface.h"

QString g_interfaceDirectory;
QString g_iconDirectory;

using namespace std;


/*
	fonctions globales
*/
bool rm(const QString& directory) {
//supprime le dossier ; il faut avoir les droits du répertoire parent
	if (!QDir(directory).exists()) return true;
	QString repertoire = directory;
	if (repertoire.right(1)!=QString("/")) repertoire += QString("/");
	QDir dir(repertoire);
	QStringList lfiles = dir.entryList(QDir::Files);
	QStringList ldirs = dir.entryList(QDir::AllDirs);
	for (int i=0; i<lfiles.count(); i++) {
		//bool b = dir.remove(lfiles.at(i));
		//if (!b) return b;
		deleteFile(repertoire+lfiles.at(i));
	}
	for (int i=2; i<ldirs.count(); i++) {
		bool b = rm(repertoire+ldirs.at(i));
		if (!b) return b;
	}
	bool b = dir.cdUp();
	if (!b) return b;
	return dir.rmdir(repertoire.section("/",-2,-2));	
}

void deleteFile(const QString& file, bool dir) {
	 if (!dir && QFile(file).exists()) QFile(file).remove();
	 else if (dir && QDir(file).exists()) rm(file);
}

bool checkPath(const QString& path) {
//vérifie les accents et caratères spéciaux du chemin donné
	//if (!QDir(path).exists() && !QFile(path).exists()) return false;
	for (int i=0; i<path.count();i++) {
		QString ss = path.left(i+1).right(1);
		if (ss==QString("/") || ss==QString("-") || ss==QString(".")) continue;
		#if defined Q_WS_WIN
			if (ss==QString(":") || ss==QString("\\")) continue;
		#endif
		QChar::Category c = path.at(i).category();
		if (c==QChar::Separator_Space || c==QChar::Punctuation_Other || c==QChar::Symbol_Other || c==QChar::Number_Other || c==QChar::Letter_Other) return false;
	}
	return true;
}

int execute(QString commande) {	//équivaut à system(ch(commande)) ; mais il y a un pb de buffer : la commande est coupée et l'exécution plante (pas toujours de pb avec la fct execute)
	QByteArray ba;
	ba.append(commande);
	return system_call(ba.constData());
}

/*
	variable globales
*/
QString applicationPath() {
//chemin de l'application ou du paquet pour MAC
	QString s = QApplication::applicationDirPath();
	#if defined Q_WS_MAC
		s = s.section("/",0,-4);
	#endif
	return s;	
}
QString dirBin(const QString& micmacdir) {
	#if defined Q_WS_WIN
		//chemin pour la commande make
		return noBlank(micmacdir) + QString("bin/");
	#else
		return QString();
	#endif
}

QString systemeNumerique(QString& virgule) {
//format du séparateur de décimales
	float a,b;
	int n = sscanf("2.5 2.6", "%f %f", &a, &b);	//fonction utilisée par elise pour parser les Pt2dr dans les xml
	if (n==2) virgule = QString(".");
	else {
		n = sscanf("2,5 2,6", "%f %f", &a, &b);
		if (n==2) virgule = QString(",");
		else return conv(QObject::tr("Fail to recognize decimal separator."));
	}
	return QString();
}
QString systemeNumerique(QString& virgule, QString& point) {
//format du séparateur de décimales
	QString err = systemeNumerique(virgule);
	if (!err.isEmpty()) return err;
	if (virgule==QString(",")) point = QString(".");
	else point = QString(",");	
	return QString();
}

/*	conversion des chaînes de caractères :
		- pour traduire : tr(char*) (ou QObject::tr)
		- tr restitue bien les accents dans les cout
		- pour restituer les accents dans les widgets (QMessageBox, QLabel, QMenu...), utiliser QApplication::translate = conv
		- traduction + accents dans les widgets : conv(tr(char*))
*/
const char* ch(const QString& s) { return s.toStdString().c_str(); }	//QString -> char* (pour les cout)
QString conv(const QString& s) { return QApplication::translate("Dialog", s.toStdString().c_str(), 0, QApplication::CodecForTr); }	//restitution des accents (peut s'utiliser avec un QObject::tr)
QString conv(const char* c) { return QApplication::translate("Dialog", c, 0, QApplication::CodecForTr); }	//restitution des accents (peut s'utiliser avec un QObject::tr)
//void qMessageBox(QWidget* w, QString title, QString msg) { QMessageBox::about(w, conv(title), conv(msg)); }	//QMessageBox avec restitution des accents
void qMessageBox(QWidget* w, QString title, QString msg) { QMessageBox::about(w, title, msg); }	//QMessageBox
ostream& ecr(const QString& s) {	//cout << s
	cout << s.toStdString() << endl;
	return cout;
}
QString remplace(const QString& s, const QString& before, const QString& after) {
	QString s2 = s;
	return s2.replace(before,after);
}
QString comm(const QString& s) {
	#if defined Q_WS_WIN
		//chemin avec "\" pour la commande cd
		return remplace(s,"/","\\");
	#else
		return s;
	#endif
}

int killall(const char* programme) {
	#if (defined Q_WS_X11 || defined Q_WS_WIN)
		if (system((string("ps -s | grep ")+string(programme)+string(" | awk '{print $1}' >tempo")).c_str())!=0) {
			cout << QObject::tr("Fail to get process PID list.").arg(programme).toStdString() << endl;
			return -1;
		}
		
		FILE* fichier = NULL;
		fichier = fopen("tempo", "r");
		char chaine[5] = "";	
		if (fichier != NULL) {
			while (fgets(chaine,5,fichier)!=NULL) {
				if (system((string("kill ")+string(chaine)).c_str())!=0) {
					cout << QObject::tr("Fail to stop process n°%1.").arg(chaine).toStdString() << endl;
					//return -1;
				}
			}
			fclose(fichier);
		}
		
		if(remove("tempo" )!=0) {
			cout << QObject::tr("Fail to delete file tempo.").toStdString() << endl;
			return -1;
		}
		return 0;
	#else
		return execute(QString("killall %1").arg(programme));
	#endif
}

void print(const QString& s) {
	#if defined Q_WS_WIN
		qDebug() << QApplication::translate("Dialog", s.toStdString().c_str(), 0, QApplication::CodecForTr);
	#else
		cout << QApplication::translate("Dialog", s.toStdString().c_str(), 0, QApplication::CodecForTr).toStdString().c_str() << endl;
	#endif
}
QString noBlank(const QString& s) { return remplace(s," ","\\ "); }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int Interface::cpu_count = 2;	//par défaut

Interface::Interface(QSettings& globParam):
	assistant( new Assistant() ),
	settings( &globParam ),
	paramMain( ParamMain() )
{
	cout << tr("Initializing GUI...").toStdString() << endl;
	paramMain.setFrench(settings->value("langue").toString()==QLocale::languageToString(QLocale::French));
	//aide
	assistant->setPages(settings->value("langue").toString()==QLocale::languageToString(QLocale::French));
	//environnement
	#if defined Q_OS_LINUX 
		cout << tr("Operating system : Linux").toStdString() << endl;	
	#endif
	#if defined Q_WS_WIN 
		cout << tr("Operating system : Windows").toStdString() << endl;		
	#endif
	#if defined Q_WS_MAC
		cout << tr("Operating system : Mac").toStdString() << endl;	
	#endif
	if (!checkPath(applicationPath())) {
		qMessageBox(this, conv(tr("Execution error")), conv(tr("Application full file path contains special caracters ; this can cause issues with some calculations. Please modify parent directory name.")));
	}

	QWidget *widget = new QWidget;
	setCentralWidget(widget);
	//couleur de fond
	QPalette palette = widget->palette();
	QColor color = palette.color(QPalette::Window);
	palette.setColor(QPalette::Base, color);
	widget->setPalette(palette);

	//élément de la fenêtre
	imagesList = new QTreeWidget;
	imagesList->move(50,60);
	imagesList->setSelectionMode (QAbstractItemView::ExtendedSelection);
	imagesList->setColumnCount(2);
	imagesList->setHeaderLabels (QStringList(tr("Images"))<<tr("Calibration"));
	imagesList->setPalette(QPalette(QColor(255,255,255)));
	//imagesList->setFrameStyle(QFrame::NoFrame);

	chantierLabel = new QLabel();
	chantierLabel->setAlignment(Qt::AlignLeft);
	chantierLabel->setTextInteractionFlags(Qt::NoTextInteraction);
	chantierLabel->setFixedHeight(20);

	pastisLabel = new QLabel();
	pastisLabel->setAlignment(Qt::AlignLeft);
	pastisLabel->setTextInteractionFlags(Qt::NoTextInteraction);
	pastisLabel->setFixedHeight(20);

	aperoLabel = new QLabel();
	aperoLabel->setAlignment(Qt::AlignLeft);
	aperoLabel->setTextInteractionFlags(Qt::NoTextInteraction);
	aperoLabel->setFixedHeight(20);

	micmacLabel = new QLabel();
	micmacLabel->setAlignment(Qt::AlignLeft);
	micmacLabel->setTextInteractionFlags(Qt::NoTextInteraction);
	micmacLabel->setFixedHeight(20);

	infoLabel = new QLabel();
	infoLabel->setAlignment(Qt::AlignLeft);
	infoLabel->setTextInteractionFlags(Qt::NoTextInteraction);
	infoLabel->setFixedHeight(100);

	QVBoxLayout *layout = new QVBoxLayout;
	layout->setMargin(5);
	layout->insertSpacing(0,20);
   	layout->addWidget(imagesList,0,Qt::AlignTop);
	layout->insertSpacing(2,20);
    	layout->addWidget(chantierLabel,0,Qt::AlignTop);
    	layout->addWidget(pastisLabel,0,Qt::AlignTop);
    	layout->addWidget(aperoLabel,0,Qt::AlignTop);
    	layout->addWidget(micmacLabel,0,Qt::AlignTop);
	layout->insertSpacing(7,20);
    	layout->addWidget(infoLabel,0,Qt::AlignTop);
    	layout->addStretch();
	
	widget->setLayout(layout);

	//titres et logos
	QWidget* topWidget = new QWidget(this);
	QGridLayout *topLayout = new QGridLayout;
	topWidget->setLayout(topLayout);
	topLabel = new QLabel;
		//logo
	topLogos = new QToolButton;
	QImage image(QString(g_iconDirectory+"LOGO_MATIS.gif"));
	image = image.scaled(50,50,Qt::KeepAspectRatio);
	topLogos->setIconSize(image.size());
	topLogos->setIcon(QPixmap::fromImage(image));
	topLogos->adjustSize();
	topLogos->setCheckable(false);
	menuBarre = new QMenuBar;
    	//topLayout->addWidget(topLabel,1,0,1,1,Qt::AlignHCenter);
    	topLayout->addWidget(topLogos,0,1,2,1,Qt::AlignRight | Qt::AlignTop);
        #ifndef Q_WS_MAC
    		topLayout->addWidget(menuBarre,0,0,1,1,Qt::AlignLeft);
	#endif
	topLogos->setPalette(QPalette(QColor(255,255,255)));
        #if defined Q_WS_MAC
		setMenuBar(menuBarre);
                layout->insertWidget(0,topLogos,0,Qt::AlignRight);
	#else
		setMenuWidget(topWidget);
	#endif
	setWindowTitle(tr("MICMAC INTERFACE"));

	//barre de menus
	createActions();
	createMenus();	

	//initialisation des paramètres globaux
	interfPastis = 0;
	interfApero = 0;
	interfMicmac = 0;
	vueHomologues = 0;
	vueChantier = 0;
	vueCartes = 0;
	interfCartes8B = 0;
	interfModele3D = 0;
	interfOrtho = 0;
	interfOptions = 0;
	interfVerifMicmac = 0;
	appliThread = 0;
	progressBar = new Progression;
	initialisation();
	if (!checkMicmac()) {
		//close();
		//return;
	}

	//taille de la fenêtre
	resize(sizeHint());
	setMinimumSize(minimumSizeHint());
	setMaximumSize(maximumSizeHint());
	QSize s = QApplication::desktop()->availableGeometry().size()/2 -sizeHint()/2;
	move( s.width(), s.height() );
	statusBar()->showMessage(conv(tr("Ready")));
	cout << conv(tr("Ready")).toStdString() << endl;	
}
Interface::~Interface() {
		delete fileMenu;
		delete calcMenu;
		delete convertMenu;
		delete helpMenu;
		delete openCalcAct;
		delete openImgAct;
		delete saveCalcAct;
		delete saveCalcAsAct;
		delete exitAct;
		delete calcPastisAct;
		delete calcAperoAct;
		delete calcMicmacAct;
		delete continueCalcAct;
		delete supprImgAct;
		delete vueHomolAct;
		delete vueAct;
		delete prof8BAct;
		delete mod3DAct;
		delete orthoAct;
		delete helpAct;
		delete aboutAct;
		delete optionAct;
    		delete chantierLabel;	
    		delete pastisLabel;	
    		delete aperoLabel;	
    		delete micmacLabel;	
    		delete infoLabel;
    		delete imagesList;
    		delete progressBar;
		delete timer;
		delete interfPastis;
		delete interfApero;
		delete interfMicmac;
		delete vueChantier;
		delete vueCartes;
		delete interfCartes8B;
		delete interfModele3D;
		delete interfOrtho;
		delete interfOptions;
		delete interfVerifMicmac;
		delete assistant;
}

int Interface::getCpuCount() { return cpu_count; }

QSize Interface::sizeHint () {
	QSize size = QApplication::desktop()->availableGeometry().size()/2;
	imagesList->setColumnWidth(0,size.width()/2);
	return size;
}

QSize Interface::minimumSizeHint () {
	return QApplication::desktop()->availableGeometry().size()/2;
}

QSize Interface::maximumSizeHint () {
	return QApplication::desktop()->availableGeometry().size();
}

void Interface::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event);
	imagesList->setColumnWidth(0,size().width()/2);
}

void Interface::initialisation() {
	defaultDir = QDir::home().absolutePath();
	if (interfPastis!=0) {
		delete interfPastis;
		interfPastis = 0;
	}
	if (interfApero!=0) {
		delete interfApero;
		interfApero = 0;
	}
	if (interfMicmac!=0) {
		delete interfMicmac;
		interfMicmac = 0;
	}
	if (vueHomologues!=0) {
		delete vueHomologues;
		vueHomologues = 0;
	}
	if (vueChantier!=0) {
		delete vueChantier;
		vueChantier = 0;
	}
	if (vueCartes!=0) {
		delete vueCartes;
		vueCartes = 0;
	}
	if (interfCartes8B!=0) {
		delete interfCartes8B;
		interfCartes8B = 0;
	}
	if (interfModele3D!=0) {
		delete interfModele3D;
		interfModele3D = 0;
	}
	if (interfOrtho!=0) {
		delete interfOrtho;
		interfOrtho = 0;
	}
	if (interfOptions!=0) {
		delete interfOptions;
		interfOptions = 0;
	}
	if (interfVerifMicmac!=0) {
		delete interfVerifMicmac;
		interfVerifMicmac = 0;
	}
	paramMain.init();

	//mise à jour de l'interface
	imagesList->clear ();
	saved = true;
	updateInterface(ParamMain::BeginMode);
}

bool Interface::checkMicmac() {
	//Vérification des droits du dossier parent (problème ../TMP pour MyRename)
	QDir parentDir = QDir::current();
	if (!parentDir.cdUp()) {
		qMessageBox(this, tr("Directory error") ,conv(tr("The parent directory of terminal directory %1 is uneditable.")).arg(QDir::currentPath()));
		return false;
	}
	if (QDir("../TMP").exists()) {
		bool b1 = rm(QString("../TMP"));
		if (!b1) {
			qMessageBox(this, tr("Directory error") ,tr("Fail to remove directory ../TMP.").arg(parentDir.canonicalPath()));
			return false;
		}
	}
	QDir dir("./");
	bool b = dir.cdUp();
	if (!b) {
		qMessageBox(this, tr("Directory error"), conv(tr("You do not have access rights for directory %1 (terminal parent directory)")).arg(parentDir.canonicalPath()));
		return false;
	}

	//récupération du nombre de processeurs	de la machine	
		//Windows
	#if defined Q_WS_WIN 
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		cpu_count=si.dwNumberOfProcessors;
	#endif
		//linux	
	#if defined Q_OS_LINUX
		FILE* fichier = NULL;
		fichier = fopen("/proc/cpuinfo", "r");
		char chaine[100] = "";				
		if (fichier != NULL) {
			while (fgets(chaine, 100, fichier) != NULL) {
				QString qs(chaine);
				if (qs.contains(QString("cpu cores"))) {	//et pour les dual cores ?
					qs = qs.simplified().section("cpu cores",-1,-1);
					for (int i=0; i<qs.size(); i++) {
						if (qs.at(i).isDigit()) {
							cpu_count = qs.left(i+1).right(1).toInt();
							break;
						}
					}
					break;
				}
			}
			fclose(fichier);
		}
	#endif	
                //mac
        #if defined Q_WS_MAC
              size_t size=sizeof(cpu_count) ;
              if (sysctlbyname("hw.ncpu",&cpu_count,&size,NULL,0)) cpu_count = 2;
        #endif
	maxcpu = (settings->value("cpu").toString().isEmpty())? cpu_count : settings->value("cpu").toInt();
	settings->setValue("cpu", maxcpu);

	//répertoire micmac
	QString micmacPath( MMDir().c_str() );
	paramMain.setMicmacDir(micmacPath);	//NB : s'il existe, le dossier micmac est validé même s'il n'y a pas tous les binaires ou la bonne configuration
	cout << tr("Micmac directory : ").toStdString(); cout << micmacPath.toStdString() << endl;

	//configuration de micmac
	bool b2 = true;	//si false, il faut réécrire le fichier MicMacConfig.xml
	int cpuLu;
	
	if (!Options::readMicMacInstall(micmacPath, cpuLu)) b2 = false;	//pas de fichier MicMacConfig.xml reconnu ou fichier erronné (chemin micmac)
	if (b2 && settings->value("cpu").toString().isEmpty() && cpuLu<=cpu_count) maxcpu = cpuLu;	//pas de cpu initial -> on prend celui de micmac
	else if (b2 && maxcpu!=cpuLu) b2 = false;	//il y a un cpu initial ou bien le cpu de micmac n'est pas possible -> on corrige le cpu de micmac
	if (!b2) Options::writeMicMacInstall(micmacPath, maxcpu);
	cout << tr("Number of usable processors : ").toStdString(); cout << maxcpu << endl;

	//vérification de la présence des fichiers exécutables requis
	QStringList err = Options::checkBinaries(micmacPath);
	if (err.count()>0) {
                for (int i=0; i<err.count(); i++) {
                        #if defined Q_WS_MAC
                               qMessageBox(this, tr("Directory error"), err.at(i)+conv(tr(" Select micmac directory using InterfaceMicmac -> Preferences -> Binaries directory micmac")));
                        #else
                               qMessageBox(this, tr("Directory error"), conv(tr("%1 Select micmac directory using Help -> Options -> Binaries directory micmac")).arg(err.at(i)));
                        #endif
                }
                return false;
	}

	return true;
}

QString Interface::pathToMicMac (QString currentPath, const QStringList& l, QString lastDir) {
	int i = l.indexOf("micmac");
	while (i!=-1) {
		//on vérifie que c'est le bon
		QString micmacDir = currentPath + l.at(i) + QString("/");
		if ( QFile(micmacDir+QString("bin")).exists() )
			return micmacDir;
		//y en a-t-il d'autres dans la liste ?
		i = l.indexOf("micmac",i+1);
	}
	//on regarde les sous-répertoires
	for (int j=0; j<l.count(); j++) {
		if (l.at(j)==QString(".") || l.at(j)==QString("..")) continue;
		QString dir = currentPath + l.at(j) + QString("/");
		if (dir==lastDir) continue;
		QStringList sl = QDir(dir).entryList(QDir::AllDirs);
		QString micmacDir = pathToMicMac (dir, sl, lastDir);
		if (!micmacDir.isEmpty())
			return micmacDir;
	}
	return QString();
}

void Interface::contextMenuEvent(QContextMenuEvent *event)
{
	if (!imagesList->geometry().contains(imagesList->parentWidget()->mapFrom(this,event->pos()))) return;
	QMenu menu(this);
	menu.addAction(supprImgAct);
	menu.exec(event->globalPos());
}

int Interface::dispMsgBox(const QString& info, const QString& question, QVector<int> reponses, int defaut)
{
	QMessageBox msgBox;
	msgBox.setText(info);
	msgBox.setInformativeText(question);

	QVector<QPushButton*> boutons(3, 0);
	boutons[0] = new QPushButton(tr("Yes"));
	boutons[1] = new QPushButton(tr("No"));
	boutons[2] = new QPushButton(tr("Cancel"));
	for (int i=0; i<3; i++) {
		switch (reponses.at(i)) {
			case 0 : msgBox.addButton(boutons[i],QMessageBox::AcceptRole);	//reponse 0
				break;
			case 1 : msgBox.addButton(boutons[i],QMessageBox::RejectRole);	//reponse 1
				break;
			case 2 : msgBox.addButton(boutons[i],QMessageBox::DestructiveRole);	//reponse 2
				break;
			default : break;
		}
		if (defaut==i) msgBox.setDefaultButton(boutons[i]);
	}
	int reponse = msgBox.exec();
	for (int i=0; i<3; i++)
		delete boutons[i];
	return reponse;
}

void Interface::openCalc(const QString& fichier)
{
	QString fichierXML;
	bool b = false;
	if (!fichier.isEmpty()) {
		if (!QFile(fichier).exists())
			qMessageBox(this, tr("File opening error.") , tr("File %1 does not exist.").arg(fichier));
		else {
			b = true;
			fichierXML = fichier;
		}
	}

	if (!b) {
		if ((!saved) && (paramMain.getCurrentMode()!=ParamMain::BeginMode)) {
			int reponse = dispMsgBox(conv(tr("Project has been modified.")), conv(tr("Do you want to save its parameters ?")), QVector<int>()<<0<<1<<2, 0);
			if (reponse == 2) {	//annuler
				return;
			}
			else if (reponse == 0) {	//oui
				saveCalc();
			}	//sinon non
		}

		//boîte de dialogue
		FileDialog fileDialog(this, tr("Open a computation"), defaultDir, tr("XML files (*.xml)") );
		fileDialog.setFileMode(QFileDialog::ExistingFile);
		fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
		QStringList fileNames;
		if (fileDialog.exec())
			fileNames = fileDialog.selectedFiles();
		else return;
	  	if (fileNames.size()==0)
			return;
		fichierXML = fileNames.at(0);
	}

	//récupération des paramètres
	initialisation();
	paramMain.setCalculXML(fichierXML.section("/",-1,-1));
	paramMain.setDossier(QDir(fichierXML.section("/",0,-2)).absolutePath()+QString("/"));
	QString lecture = ParamCalcul::lire(paramMain, fichierXML );
	if(!lecture.isEmpty()) {
		qMessageBox(this, tr("File opening error.") ,lecture);
		return;
	}
	updateInterface(paramMain.getCurrentMode());

	QTreeWidgetItem* treeWidgetItem = new QTreeWidgetItem(QStringList(paramMain.getDossier())<<QString());
	imagesList->addTopLevelItem(treeWidgetItem);

	//affichage de la liste d'images
	if (paramMain.getCurrentMode()!=ParamMain::BeginMode) {
		QVector<ParamImage>* correspImgCalib = &(paramMain.modifCorrespImgCalib());
		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
			//récupération des images tif
			QString lecture = FichierParamImage::lire(paramMain.getDossier()+paramMain.getImageXML(), *correspImgCalib);
			if(!lecture.isEmpty()) {
				qMessageBox(this, tr("Images read error") ,paramMain.getDossier()+paramMain.getImageXML()+"\n"+lecture);
				return;
			}
			//numéro des images
			if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::PtsInteret)
				paramMain.calcImgsId();
			//application aux images orientables
			/*if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours) {
				for (int i=0; i<paramMain.getCorrespImgCalib().count(); i++) {
					int index = paramMain.getParamApero().getImgToOri().indexOf(paramMain.getCorrespImgCalib().at(i).getImageRenamed());
					if (index!=-1)
						paramMain.modifParamApero().modifImgToOri()[index] = paramMain.getCorrespImgCalib().at(i).getImageTif();
				}
			}*/
			//récupération des calibrations
			if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::PtsInteret) {
				lecture = FichierAssocCalib::lire(paramMain.getDossier()+paramMain.getAssocCalibXML(), *correspImgCalib);
				if(!lecture.isEmpty()) {
					qMessageBox(this, tr("Calibrations read error.") ,lecture);
					return;
				}
			}

			if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::PtsInteret) {
				QString err = paramMain.saveImgsSize();
				if(!err.isEmpty()) {
					qMessageBox(this, tr("Read error.") ,err);
					return;
				}
			}

			if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours && paramMain.getCurrentMode()!=ParamMain::PointsMode) {
				QString err = paramMain.saveImgsSize();	//pour pouvoir utiliser convert
				if (!err.isEmpty()) {
					qMessageBox(this, tr("Read error."), err);
					return;
				}

				//récupération de l'image maîtresse
				QString maitresse;
				lecture = FichierMaitresse::lire(paramMain.getDossier()+paramMain.getMaitresseXML(), maitresse);
				if(!lecture.isEmpty()) {
					qMessageBox(this, tr("Read error."), lecture);
					return;
				}
				if (QFile(paramMain.getDossier()+maitresse).exists())
					paramMain.modifParamApero().setImgMaitresse(maitresse);

				//récupération des images orientables
				/*QStringList* imgOri = new QStringList;
				lecture = FichierImgToOri::lire(paramMain.getDossier()+paramMain.getImgOriXML(), imgOri, correspImgCalib);
				if(!lecture.isEmpty()) {
					qMessageBox(this, tr("Read error."), lecture);
					return;
				}
				imgOri->push_back(paramMain.getImgMaitresse());
				paramMain.setImgOri(imgOri);	*/

				/*if (paramMain.getCurrentMode()==ParamMain::CarteEnCours || paramMain.getCurrentMode()==ParamMain::EndMode) {
					//récupération des masques
					QString masque, refmasque;
					lecture = FichierDefMasque::lire(paramMain.getDossier(),paramMain.getDefMasqueXML(), masque, refmasque);
					if(!lecture.isEmpty()) {
						qMessageBox(this, tr("Read error.") ,lecture);
						return;
					}
					paramMain.setMasque(masque);
					paramMain.setRefMasque(refmasque);
				}*/
			}
		}
		for (QVector<ParamImage>::const_iterator it=paramMain.getCorrespImgCalib().begin(); it!=paramMain.getCorrespImgCalib().end(); it++) {
			QTreeWidgetItem* treeWidgetItem;
			if (paramMain.getCurrentMode()==ParamMain::ImageMode || ( paramMain.getCurrentMode()==ParamMain::PointsEnCours && paramMain.getAvancement()==PastisThread::Enregistrement))
				treeWidgetItem = new QTreeWidgetItem(QStringList(it->getImageRAW())<<QString());
			else if (paramMain.getCurrentMode()==ParamMain::PointsEnCours && (paramMain.getAvancement()==PastisThread::Conversion || paramMain.getAvancement()==PastisThread::Ecriture))
				treeWidgetItem = new QTreeWidgetItem(QStringList(it->getImageTif())<<QString());
			else
				treeWidgetItem = new QTreeWidgetItem(QStringList(it->getImageTif())<<it->getCalibration());
			imagesList->topLevelItem(0)->addChild(treeWidgetItem);
		}	//NB : le fichier de calcul a forcément un dossier (pas d'enregistrement dans BeginMode)
		imagesList->expandAll();
		imagesList->adjustSize();
		adjustSize();
	}
	if (paramMain.getCurrentMode()==ParamMain::CarteEnCours || paramMain.getCurrentMode()==ParamMain::EndMode) {
		for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
			CarteDeProfondeur* carte = &paramMain.modifParamMicmac()[i];

			bool ok;
			QString numCarte = paramMain.getNumImage(carte->getImageDeReference(),&ok,false);
			if (!ok) {
				qMessageBox(this, tr("Read error"), conv(tr("Image %1 number is uncorrect.")).arg(carte->getImageDeReference()));				
				return;
			}
			//vérification du fichier de référencement
			if (!QFile(carte->getReferencementMasque(paramMain)).exists()) {
				qMessageBox(this, tr("Read error"), conv(tr("Mask referencing file %1 not found.")).arg(carte->getReferencementMasque(paramMain)));			
				return;
			}
			//orthoimages déjà calculées
			if (paramMain.getCurrentMode()==ParamMain::EndMode) {
				QString dossierOrtho = paramMain.getDossier() + QString("ORTHO%1/").arg(numCarte);
				if (QFile(dossierOrtho+QString("Ortho-NonEg-Test-Redr.tif")).exists() || QFile(dossierOrtho+QString("Ortho-Eg-Test-Redr.tif")).exists())
					carte->setOrthoCalculee(true);
				else carte->setOrthoCalculee(false);
			}
		}
	}
}

void Interface::openImg()
{
	if ((!saved) && (paramMain.getCurrentMode()!=ParamMain::BeginMode) && (paramMain.getCurrentMode()!=ParamMain::ImageMode)) {
		int reponse = dispMsgBox(conv(tr("Project has been modified.")), conv(tr("Do you want to save its parameters ?")), QVector<int>()<<0<<1<<2, 0);
		if (reponse == 2) {	//annuler
			return;
		}
		else if (reponse == 0) {	//oui
			saveCalc();
		}	//sinon non
		paramMain.setDossier(QString());
	}
	if ((paramMain.getCurrentMode()!=ParamMain::BeginMode) && (paramMain.getCurrentMode()!=ParamMain::ImageMode)) initialisation();

	//boîte de dialogue
	QString dir = (paramMain.getDossier().isEmpty())? defaultDir : paramMain.getDossier();
	QString acceptedFormats;
	for (int i=0; i<paramMain.getFormatsImg().count(); i++)
		acceptedFormats += tr("Images (*.%1);;").arg(paramMain.getFormatsImg().at(i));
	acceptedFormats += tr("Images (*.tif);;Images (*.tiff);;");
	acceptedFormats += tr("All images (*);;");
	FileDialog fileDialog(this, tr("Load images"), dir, acceptedFormats );
	fileDialog.setFileMode(QFileDialog::ExistingFiles);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec())
		fileNames = fileDialog.selectedFiles();
	else return;
  	if (fileNames.size()==0)
		return;
	int n=0;
	if (imagesList->topLevelItemCount()!=0)
		n = imagesList->topLevelItem(0)->childCount();

	//on vérifie que le nom du dossier est lisible (à cause des accents)
	if (!checkPath(fileNames.at(0))) {
		qMessageBox(this, tr("Read error"), conv(tr("Fail to read directory.\nCheck there are no accents in path.")));	
		return;
	}

	//lecture et affichage des images
	for (QStringList::const_iterator it=fileNames.begin(); it!=fileNames.end(); it++) {
		QString s = *it;

		//on vérifie que le nom du fichier est lisible (à cause des accents)
		if (!checkPath(s)) {
			qMessageBox(this, tr("Read error"), conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(s));	
			return;
		}

		//extraction du dossier et du nom de l'image
		if (it==fileNames.begin()) {
			QString s2 = s.section("/",0,-2);


			s2 = QDir(s2).absolutePath() + QString("/");
			if (!(paramMain.getDossier().isEmpty()) && (s2!=paramMain.getDossier())) {
				int reponse2 = dispMsgBox(conv(tr("The data directory must be the same for all images.")), conv(tr("Do you want to remove previously loaded images ?")), QVector<int>()<<0<<-1<<1, 2);	//supprimer les images ?
				if (reponse2 == 1)	//annuler
					return;
				else {	//oui
					initialisation();
					n = 0;
				}
			}
			paramMain.setDossier(s2);
			if (imagesList->topLevelItemCount()==0) {	//nouveau dossier, nouvelle liste
				QTreeWidgetItem* treeWidgetItem = new QTreeWidgetItem(QStringList(paramMain.getDossier())<<QString());
				imagesList->addTopLevelItem(treeWidgetItem);
			}
		}
		QString img = s.section('/', -1, -1);

		//si c'est une image tif couleur, on prend l'image N&B
		if (img.right(12)==QString("_couleur.tif") && QFile(paramMain.getDossier()+img.left(img.count()-12)+QString(".tif")).exists()) img = img.left(img.count()-12)+QString(".tif");

		//on vérifie que l'image n'a pas déjà été ajoutée
		int i=0;
		if (n>0) {
			while (i<n && img!=imagesList->topLevelItem(0)->child(i)->text(0))
				i++;
		}
		if (i<n) continue;

		//enregistrement et affichage
		QTreeWidgetItem* treeWidgetItem = new QTreeWidgetItem(QStringList(img)<<QString());
		imagesList->topLevelItem(0)->addChild(treeWidgetItem);
		n++;
		ParamImage array;
			array.setImageRAW(img);
			array.setImageTif(QString());
		QString extension = img.section(".",-1,-1);
	/*	if (img.section(".",-1,-1).toUpper()==QString("tiff").toUpper() || img.section(".",-1,-1).toUpper()==QString("tif").toUpper()) {	//on renomme les .tiff en .tif et en minuscules
			QFile(paramMain.getDossier()+img).rename(paramMain.getDossier()+img.section(".",0,-2)+QString(".tif"));
			img = img.section(".",0,-2)+QString(".tif");
		}*/
		/*if (img.section(".",-1,-1)==QString("tif")) {
			array.setImageRenamed(img);
			array.setImageTif(img);
		}*/
		paramMain.modifCorrespImgCalib().push_back(array);
	}
	imagesList->expandAll();
	imagesList->adjustSize();
	adjustSize();

	saved=false;
	updateInterface(ParamMain::ImageMode);			
}

void Interface::saveCalcSsMsg() { saveCalc(paramMain.getDossier()+QString("sauvegarde.xml"),false); }

void Interface::saveCalc(QString file, bool msg)
{
	if (paramMain.getCalculXML().isEmpty() && file.isEmpty()) {
		saveCalcAs();
		return;
	}
	QString sauvegarde = (file.isEmpty())? paramMain.getCalculXML() : file;

	if(!(ParamCalcul::ecrire(paramMain,sauvegarde))) {
		qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save compution.")));
		return;
	}
	if (file.isEmpty() && msg) {
		qMessageBox(this, tr("Save"), conv(tr("Compution saved.")));
	}
	saved=true;
}

void Interface::saveCalcAs()
{	
	QString dir = (paramMain.getDossier().isEmpty())? defaultDir : paramMain.getDossier();
	QString fileName = FileDialog::getSaveFileName(this,tr("Save"), dir,tr("XML files (*.xml);;all files (*)"));
	if (fileName.isEmpty())
		return;

	QString nom = (fileName.contains("."))? fileName.section(".",0,-2) : fileName;
	QString extension = (fileName.contains("."))? fileName.section(".",-1,-1) : QString();
	if (extension!=QString("xml"))
		fileName = nom + QString(".xml");
	paramMain.setCalculXML(fileName);
	saveCalc();
}

bool Interface::closeAppli()
{
	static int closetime = 0;
	if (closetime!=0) return true;	//sinon appelle closeEvent qui appelle closeAppli
	if (!saved) {
		int reponse = dispMsgBox(conv(tr("Project has been modified.")), conv(tr("Do you want to save its parameters ?")), QVector<int>(3)<<0<<1<<2, 0);
		if (reponse == 2) {	//annuler
			return false;
		}
		else if (reponse == 0) {	//oui
			saveCalc();
		}	//sinon non
	}
	closetime++;
	close();	//appelle au closeEvent mais le "close" courant est traité avant et supprime la liste des événements
	return true;
}
void Interface::closeEvent(QCloseEvent* event) {
	if (closeAppli()) event->accept();
	else event->ignore();
}

void Interface::calcPastis(bool continuer)
{
	//-- bool refaire = false;
	if (!continuer) {
		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
			int reponse = dispMsgBox(conv(tr("Tie points were already found.")), conv(tr("Do you want to remove them from the calculation ?")), QVector<int>()<<0<<-1<<1, 2);
			if (reponse == 1) {	//annuler
				return;
			}	//sinon oui
			//-- refaire = true;	//des paramètres ont déjà été saisis (dans cette session ou ouverture d'un calcul)
		}
		if (imagesList->topLevelItem(0)->childCount()<2) {	
			qMessageBox(this, conv(tr("Parameter error.")), conv(tr("Not enough images.")));
			return;
		}
		paramMain.setAvancement(PastisThread::Enregistrement);
		updateInterface(ParamMain::ImageMode);

		//ouverture de la fenêtre des paramètres
		if (interfPastis==0) {	//la fenêtre n'a pas encore été ouverte dans cette session
			interfPastis = new InterfPastis(this, assistant, &paramMain);
			if (!interfPastis->isDone()) {
				interfPastis = 0;
				return;
			}
		}
		interfPastis->show();
		int rep = interfPastis->exec();
		if (rep != QDialog::Accepted)
			return;
		
		//enregistrement des paramètres (hors du calcul par thread, les cpls ne sont pas enregistrés autrement, on ne peut pas reprendre cette partie)
		paramMain.setParamPastis(interfPastis->getParamPastis());
			//écriture de la liste des couples d'images
		if (!FichierCouples::ecrire (paramMain.getDossier()+paramMain.getCoupleXML(), interfPastis->getParamPastis().getCouples())) {
			qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to save images pairs.")));
			return;
		}
	}

	paramMain.setCurrentMode(ParamMain::PointsEnCours);
	pastisLabel->setText(tr("Computing tie points."));

	//progress dialog
	//-- int nbCpls;
	if (interfPastis==0){
		QList<pair<QString, QString> > couples;
		QString lecture = FichierCouples::lire (paramMain.getDossier()+paramMain.getCoupleXML(), couples, paramMain.getCorrespImgCalib());
		//-- nbCpls = couples.count();
	}
	//-- else
	//--	nbCpls = interfPastis->getParamPastis().getCouples().count();

	stdoutfilename = paramMain.getDossier()+QString("pastis_outstream");
	appliThread = new PastisThread(&paramMain, stdoutfilename, &annulation, maxcpu);
	setUpCalc();
}

void Interface::calcApero(bool continuer)
{
	if (paramMain.getParamApero().getImgToOri().count()<2) {
			qMessageBox(this, conv(tr("Execution error.")), conv(tr("Not enough images to be oriented.")));
		return;
	}
	if (!continuer) {
		if (paramMain.getCurrentMode()!=ParamMain::PointsMode) {
			int reponse = dispMsgBox(conv(tr("Orientations were already computed.")), conv(tr("Do you want to remove them from the calculation ?")), QVector<int>()<<0<<-1<<1, 2);
			if (reponse == 1) {	//annuler
				return;
			}	//oui
		}
		paramMain.setAvancement(AperoThread::Enregistrement);
		updateInterface(ParamMain::PointsMode);

		//ouverture de la fenêtre des paramètres
		if (interfApero==0) {	//la fenêtre n'a pas encore été ouverte dans cette session
			QString err = paramMain.saveImgsSize();	//pour pouvoir utiliser convert
			if (!err.isEmpty()) {
				qMessageBox(this, conv(tr("Read error.")), err);
				return;
			}
			interfApero = new InterfApero(&paramMain, this, assistant);
		}
		interfApero->show();
		int rep = interfApero->exec();
		if (rep != QDialog::Accepted)
			return;

		//paramètres
		paramMain.setParamApero(interfApero->getParamApero());

		//remise à 0 de la vue des points homologues (dépend du filtrage des points)
		if (vueHomologues!=0) {
			delete vueHomologues;
			vueHomologues = 0;
		}		
	}

	paramMain.setCurrentMode(ParamMain::PoseEnCours);
	aperoLabel->setText(tr("Computing poses."));
	delete vueChantier;
	delete vueCartes;
	vueChantier = 0;
	vueCartes = 0;

	stdoutfilename = paramMain.getDossier()+QString("apero_outstream");
	appliThread = new AperoThread(&paramMain, stdoutfilename, &annulation, maxcpu);
	setUpCalc();
}

void Interface::calcMicmac(bool continuer)
{
	//-- bool refaire = false;
	if (!continuer) {
		//-- if (paramMain.getCurrentMode()!=ParamMain::PoseMode)
		//--	refaire = true;	//des paramètres ont déjà été saisis (dans cette session ou ouverture d'un calcul)
		paramMain.setAvancement(MicmacThread::Calcul);
		paramMain.setAvancement(MicmacThread::Enregistrement);
		updateInterface(ParamMain::PoseMode);

		//ouverture de la fenêtre des paramètres
		if (interfMicmac==0) {	//la fenêtre n'a pas encore été ouverte dans cette session
			int i=paramMain.getParamPastis().getTypeChantier();
			interfMicmac = new InterfMicmac(this, &paramMain, i, vueChantier, assistant);
		}
		paramMain.setCurrentMode(ParamMain::CarteEnCours);	//pour pouvoir sauvegarder les paramètres avant le lancement du calcul
		interfMicmac->show();
		int rep = interfMicmac->exec();
		if (rep != QDialog::Accepted) {
			if (QFile(applicationPath()+QString("/masquetempo.tif")).exists())
				QFile(applicationPath()+QString("/masquetempo.tif")).remove();
			if (QFile(applicationPath()+QString("/masquetemponontuile.tif")).exists())
				QFile(applicationPath()+QString("/masquetemponontuile.tif")).remove();
			return;
		}
		paramMain.setParamMicmac(interfMicmac->getParamMicmac());
		if (paramMain.getParamMicmac().count()==0) return;
	}

	//paramMain.setCurrentMode(ParamMain::CarteEnCours);
	micmacLabel->setText(tr("Computing depth maps."));
	delete vueCartes;
	vueCartes = 0;

	stdoutfilename = paramMain.getDossier()+QString("micmac_outstream");
	appliThread = new MicmacThread(&paramMain, stdoutfilename, &annulation);
	setUpCalc();
	(*verifMicmacAct).setEnabled(true);
}

void Interface::setUpCalc() {
//départ des calculs
	//progress dialog
	progress = new QProgressDialog(QString(), tr("Stop computing"), 0, progressBar->maxComplete(paramMain), this);
	progress->setWindowTitle(tr("Computing in progress"));
	progress->setWindowModality(Qt::NonModal);//Qt::ApplicationModal
	progress->setMinimumDuration(0);
	progress->setLabelText(conv(tr("User data saving")));
	connect(progress, SIGNAL(canceled()), this, SLOT(progressCanceled()));

	//calcul par thread
	annulation = false;
	connect(appliThread, SIGNAL(finished()), this, SLOT(threadFinished()));
	connect(appliThread, SIGNAL(saveCalcul()), this, SLOT(saveCalcSsMsg()));
	posReadError = 0;	

	//départ
	appliThread->start();

	//animation de la progress dialog
	timer = new QTimer;
	timer->setSingleShot(true);
	progress->setValue(0);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateProgressDialog()));
	timer->start(1000);	//en ms
}

void Interface::updateProgressDialog () {
	//animation de la ProgressDialog
	if (progress==0) return;
	if (progress->wasCanceled()) return;

	//incrémentation en fonction des fichiers créés
	progress->setLabelText(appliThread->getProgressLabel());
	if (progress->maximum()==1)
		progress->setValue(-1);	//progressBar "ping-pong"
	else
		progressBar->updatePercent(paramMain, progress, stdoutfilename);
	progress->adjustSize();

	timer->start(1000);	//en ms
}

void Interface::progressCanceled() {	//annuler
//le calcul est interrompu
	annulation = true;
	disconnect(progress, SIGNAL(canceled()), this, SLOT(progressCanceled()));
	disconnect(appliThread, SIGNAL(finished()), this, SLOT(threadFinished()));
	if (appliThread->killProcess()) appliThread->terminate();
	threadFinished();
}

void Interface::threadFinished() {	//value=maxValue
//le calcul est terminé (erreur détectée ou le calcul a abouti)
	if (progress!=0) {
		disconnect(progress, SIGNAL(canceled()), this, SLOT(progressCanceled()));
		delete progress;
		progress = 0;
	}
	saved=false;
	const QVector<ParamImage>* correspImgCalib = &(paramMain.getCorrespImgCalib());
	switch (paramMain.getCurrentMode()) {
		case ParamMain::PointsEnCours :
			if (paramMain.getAvancement()<0) {	//une erreur est survenue
				displayErreur();
				updateInterface(ParamMain::PointsEnCours);
				paramMain.setAvancement(0);	//on ne sait pas d'où vient vraiment l'erreur
				delete appliThread;
				delete timer;
				return;
			}
			//affichage des images et des calibrations internes correspondantes
			if (paramMain.getAvancement()==PastisThread::Enregistrement) {
				for (int i=0; i<correspImgCalib->size(); i++)
					imagesList->topLevelItem(0)->child(i)->setText(0,correspImgCalib->at(i).getImageRAW());
			} else {
				for (int i=0; i<correspImgCalib->size(); i++)
					imagesList->topLevelItem(0)->child(i)->setText(0,correspImgCalib->at(i).getImageTif());
			}
			if (paramMain.getAvancement()>=PastisThread::PtsInteret) {
				for (int i=0; i<correspImgCalib->size(); i++)
					imagesList->topLevelItem(0)->child(i)->setText(1,correspImgCalib->at(i).getCalibration());
			}
			imagesList->expandAll();
			imagesList->adjustSize();
			if (paramMain.getAvancement()<appliThread->getEndResult()) {	//le calcul a été stoppé en dehors d'un processus
				updateInterface(ParamMain::PointsEnCours);
				delete appliThread;
				delete timer;
				return;
			} else {	//le calcul a abouti
				updateInterface(ParamMain::PointsMode);
			}
			break;
		case ParamMain::PoseEnCours :
			if (paramMain.getAvancement()<0) {	//une erreur est survenue
				displayErreur();
				updateInterface(ParamMain::PoseEnCours);
				paramMain.setAvancement(0);	//on ne sait pas d'où vient vraiment l'erreur
				delete appliThread;
				delete timer;
				return;
			}
			if (paramMain.getAvancement()<appliThread->getEndResult()) {	//le calcul a été stoppé en dehors d'un processus
				updateInterface(ParamMain::PoseEnCours);
				delete appliThread;
				delete timer;
				return;
			}
			updateInterface(ParamMain::PoseMode);	//le calcul a abouti
			break;
		case ParamMain::CarteEnCours :
			if (paramMain.getAvancement()<0) {	//une erreur est survenue
				displayErreur();
				updateInterface(ParamMain::CarteEnCours);
				paramMain.setAvancement(0);	//on ne sait pas d'où vient vraiment l'erreur
				delete appliThread;
				delete timer;
				return;
			}
			if (paramMain.getAvancement()<appliThread->getEndResult()) {	//le calcul a été stoppé en dehors d'un processus
				updateInterface(ParamMain::CarteEnCours);
				delete appliThread;
				delete timer;
				return;
			}
			updateInterface(ParamMain::EndMode);	//le calcul a abouti
			break;
		case ParamMain::EndMode :
			if (paramMain.getAvancement()<0) {	//une erreur est survenue
				displayErreur();
				paramMain.setAvancement(0);	//on ne sait pas d'où vient vraiment l'erreur
				delete appliThread;
				delete timer;
				return;
			}
			updateInterface(ParamMain::EndMode);	//met à jour infoLabel
			break;
		default : return;
	}	
	paramMain.setAvancement(0);	//cas où le calcul a abouti
	saveCalc(paramMain.getDossier()+QString("sauvegarde.xml"));
	delete appliThread;
	delete timer;
}

void Interface::displayErreur () {
	switch (paramMain.getCurrentMode()) {
		case ParamMain::PointsEnCours :
			switch (paramMain.getAvancement()) {
				case -100 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to write makefile %1.")).arg(appliThread->getReaderror()));
					break;
				case -2 :
					qMessageBox(this, conv(tr("Read error.")), conv(tr("Image %1 focal is longer than 1 m.")).arg(appliThread->getReaderror()));
					break;
				case -3 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to convert images into tif format.")));
					break;
				case -4 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to create image sub-directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -5 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to move images %1 into sub-directory %2.")).arg(appliThread->getReaderror()).arg(appliThread->getReaderror().toUpper()));
					break;
				case -6 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save depth map list.")));
					break;
				case -7 :
					qMessageBox(this, conv(tr("Write error.")), appliThread->getReaderror());
					break;
				case -8 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save image - calibration matches.")));
					break;
				case -9 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to create makefile for tie point computing.")));
					break;
				case -10 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Tie point computation failed.\nTo get more details, see file %1.")).arg(stdoutfilename));
					break;
				case -11 :
					qMessageBox(this, tr("Read error."), appliThread->getReaderror());
					break;
				case -12 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to fill a calibration file.")));
					break;
				case -13 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to extract default parameters from a calibration file.")));
					break;
				case -16 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to read makefile for tie point computing.")));
					break;
				case -17 :
					for (int i=0; i<paramMain.getCorrespImgCalib().size(); i++) {
						imagesList->topLevelItem(0)->child(i)->setText(1,paramMain.getCorrespImgCalib().at(i).getCalibration());
					}
					qMessageBox(this, tr("Read error."), conv(tr("An image has no matching calibration.")));
					break;
				case -22 :
					qMessageBox(this, tr("Read error."), appliThread->getReaderror());
					break;
				case -23 :
					qMessageBox(this, tr("Read error."), conv(tr("Fail to read image %1.")).arg(appliThread->getReaderror()));
					break;
				case -25 :
					qMessageBox(this, tr("Read error."), conv(tr("Fail to save image pair file for computation second step.")));
					break;
				case -26 :
					qMessageBox(this, tr("Read error."), conv(tr("Fail to read image pair file.")));
					break;
				case -27 :
					qMessageBox(this, conv(tr("Write error.")), appliThread->getReaderror());
					break;
				case -28 :
					qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to create link to micmac/bin directory.")));
					break;
				case -31 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to remove directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -32 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Image %1 does not exist or does not match camera parameters.")).arg(appliThread->getReaderror()));
					break;
				case -33 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to create link to image %1.")).arg(appliThread->getReaderror()));
					break;
				case -34 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to compute image key number.")));
					break;
				case -35 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to extract image %1 metadata (size and focal length).")).arg(appliThread->getReaderror()));
					break;
				case -36 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Image %1 focal length is not managed by this GUI.")).arg(appliThread->getReaderror()));
					break;
				default : return;
			}
			break;
		case ParamMain::PoseEnCours :
			switch (paramMain.getAvancement()) {
				case -1 :
					qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to save master image.")));
					break;
				case -2 :
					qMessageBox(this, conv(tr("Write error")),conv(tr("Fail to save list of images to be oriented %1.")).arg(appliThread->getReaderror()));
					break;
				case -3 :
					qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to save calibration definition list %1.")).arg(appliThread->getReaderror()));
					break;
				case -4 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Pose %1 computing failed.\nTo get more details, refer to %2 file.")).arg(appliThread->getReaderror()).arg(stdoutfilename));
					break;
				case -5 :
				qMessageBox(this, tr("Read error."), conv(tr("Fail to recognize calibration focal length from file name.")));
					break;
				case -6 :
				qMessageBox(this, tr("File error."), appliThread->getReaderror());
					break;
				case -7 :
				qMessageBox(this, conv(tr("Execution error.")), conv(tr("Fail to create directory Ori-F/3D.")));
					break;
				case -8 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save image %1 tie-points computed in 3D.")).arg(appliThread->getReaderror()));
					break;
				case -9 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save survey parameters.")));
					break;
				case -10 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save tie-points in file Ori-F/3D/allPoints.txt.")));
					break;
				case -11 :
				qMessageBox(this, tr("Read error."), conv(tr("Fail to read image %1.")).arg(appliThread->getReaderror()));
					break;
				case -12 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to write file Filtrage.xml.")));
					break;
				case -13 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to filter tie-points.")));
					break;
				case -14 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to remove directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -15 :
					qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to rename directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -16 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save pose list %1.")).arg(appliThread->getReaderror()));
					break;
				case -17 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save calibration %1 key.")).arg(appliThread->getReaderror()));
					break;
				case -18 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save absolute orientation %1.")).arg(appliThread->getReaderror()));
				case -19 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save free image - calibration matche file.")));
					break;
				case -20 :
				qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save camera constraint file Contraintes.xml.")));
					break;
				case -21 :
				qMessageBox(this, conv(tr("Execution error")), appliThread->getReaderror());
					break;
				case -22 :
				qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to convert GCP file %1 into xml format for Apero.")).arg(appliThread->getReaderror()));
					break;
				case -23 :
				qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to convert image measurement file %1 of GCP into xml format for Apero.")).arg(appliThread->getReaderror()));
					break;
				case -24 :
				qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to convert file %1 for tie-points and camera poses export to ply format into xml format for Apero.")).arg(appliThread->getReaderror()));
					break;
				case -25 :
				qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to read calibration file %1.")).arg(appliThread->getReaderror()));
					break;
				case -26 :
				qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to save initial orientation file.")).arg(appliThread->getReaderror()));
					break;
				default : return;
			}
			break;
		case ParamMain::CarteEnCours :
			switch (paramMain.getAvancement()) {
				case -1 :
					qMessageBox(this, tr("Read error"),appliThread->getReaderror());
					break;
				case -2 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save mask definition file.")));
					break;
				case -3 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to save depth map list.")));
					break;
				case -4 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Depth map computation failed.\nTo get more details, see file %1.")).arg(stdoutfilename));
					break;
				case -5 :
					qMessageBox(this, conv(tr("Write error.")), conv(tr("Fail to write output file.")));
					break;
				case -6 :
					qMessageBox(this, tr("Read error"),conv(tr("Fail to extract referencing image %1 number.")).arg(appliThread->getReaderror()));
					break;
				case -7 :
					qMessageBox(this, tr("Read error"),conv(tr("Fail to create directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -8 :
					qMessageBox(this, conv(tr("Parameter error (internal)")),conv(tr("No depth map parameter provided to compute the IM.")));
					break;
				case -9 :
					qMessageBox(this, conv(tr("Parameter error (internal)")),conv(tr("No depth map parameter saved.")));
					break;
				case -10 :
					qMessageBox(this, conv(tr("Execution error")),conv(tr("Fail to remove directory %1.")).arg(appliThread->getReaderror()));
					break;
				case -11 :
					qMessageBox(this, conv(tr("Execution error")),conv(tr("Fail to write orthoimage parameters.")));
				case -12 :
					qMessageBox(this, conv(tr("Write error")),conv(tr("Fail to save search interval for correlation.")));
				case -13 :
					qMessageBox(this, conv(tr("Execution error")),conv(tr("Fail to save high slope control parameter.")));
					break;
				case -14 :
					qMessageBox(this, conv(tr("Write error")),conv(tr("Fail to save image list for correlation.")));
					break;
				default : return;
			}
			break;
		case ParamMain::EndMode :
			switch (paramMain.getAvancement()) {
				case -1 :
					qMessageBox(this, tr("Read error"),conv(tr("Mask %1 not found.")).arg(appliThread->getReaderror()));
					break;
				case -2 :
					qMessageBox(this, conv(tr("Execution error")),conv(tr("Fail to convert image %1.")).arg(appliThread->getReaderror()));
					break;
				default : return;
			}
			break;
		default : return;
	}
}

void Interface::continueCalc()
{
	saved=false;
	if (paramMain.getCurrentMode()==ParamMain::PointsEnCours) {
		calcPastis(true);
	} else if (paramMain.getCurrentMode()==ParamMain::PoseEnCours) {
		calcApero(true);
	} else if (paramMain.getCurrentMode()==ParamMain::CarteEnCours) {
		calcMicmac(true);
	}
}

void Interface::vueHomol()
{
//visualisation 2D des points homologues
	if (vueHomologues==0) {
		QApplication::setOverrideCursor( Qt::WaitCursor );
		vueHomologues = new VueHomologues(&paramMain, assistant, this);
		QApplication::restoreOverrideCursor();
	}
	if (!vueHomologues->getDone()) {
		return;
	}
	vueHomologues->show();
	vueHomologues->exec();
}

void Interface::vue()
{
//visualisation 3D du chantier (centre des caméras et points de liaison) avec openGL
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vueChantier==0) vueChantier = new VueChantier(&paramMain, this, assistant);
	if (!vueChantier->isDone()) {
		QApplication::restoreOverrideCursor();
		return;
	}
	vueChantier->show(SelectCamBox::Hide);
	QApplication::restoreOverrideCursor();
	vueChantier->exec();
}

void Interface::vueNuages()
{
//visualisation 3D du chantier et des nuages avec openGL
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vueCartes==0) {
		if (vueCartes==0) vueCartes = new VueChantier(&paramMain, this, assistant);
		if (!vueCartes->isDone()) {
			QApplication::restoreOverrideCursor();
			return;
		}
		GLParams* glParam = &(vueCartes->modifParams());

		//calcul des points du nuage
		glParam->modifNuages().resize(paramMain.getParamMicmac().count());
		QVector<QString> imgRef(paramMain.getParamMicmac().count());
		QVector<Model3DThread> model3DThread(paramMain.getParamMicmac().count(),Model3DThread(&paramMain, glParam));
		for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
			imgRef[i] = paramMain.getParamMicmac().at(i).getImageDeReference().section(".",0,-2);
			model3DThread[i].setI(i);
		}
		for (int i=0; i<paramMain.getParamMicmac().count(); i++)
			model3DThread[i].start();
		for (int i=0; i<paramMain.getParamMicmac().count(); i++)
			while (model3DThread[i].isRunning()) {}
		for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
			if (!model3DThread[i].getIsDone()) {
				QApplication::restoreOverrideCursor();		
				qMessageBox(this, tr("Error"),model3DThread[i].getReaderror());
				return;
			}
		}

		//affichage
	        if (!vueCartes->addNuages(imgRef, paramMain)) {
			qMessageBox(this, tr("Read error"), conv(tr("Fail to add point clouds into the 3D view")));
			return;
		}

		//maj interface
		infoLabel->setText(infoLabel->text() + conv(tr("\nPoint cloud computation completed.")));
	}
	vueCartes->show(SelectCamBox::Hide);
	QApplication::restoreOverrideCursor();
	vueCartes->exec();
}

void Interface::cartesProf8B()
{
	QList<ParamConvert8B::carte8B> cartes;
	//trouver les cartes
	for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
		//numéro de l'image de référence
		bool ok = false;
		QString numRef = paramMain.getNumImage( paramMain.getParamMicmac().at(i).getImageDeReference(), &ok, false);
		if (!ok) {
			qMessageBox(this, conv(tr("Parameter error")), conv(tr("Fail to extract reference image %1 number.")).arg(paramMain.getParamMicmac().at(i).getImageDeReference()));
			return;
		}

		QString refImg = paramMain.getParamMicmac().at(i).getImageDeReference();
		QString imgTexture;
		if (paramMain.getParamMicmac().at(i).getRepere()) imgTexture = paramMain.convertTifName2Couleur(refImg);
		else if (paramMain.getParamMicmac().at(i).getOrthoCalculee() && QFile(paramMain.getDossier()+QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(numRef)).exists()) imgTexture = QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(numRef);
		else if (paramMain.getParamMicmac().at(i).getOrthoCalculee() && QFile(paramMain.getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numRef)).exists()) imgTexture = QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numRef);
		else  imgTexture = paramMain.getParamMicmac().at(i).getImageSaisie(paramMain);	//TA n&b

		int dezoom = 32;   //se trouve à <EtapeMEC><DeZoom > dans param-GeoIm/Terrain.xml (ou param utilisateur ? -> à récupérer dans interfMicmac)
		int j = 1;
		while (dezoom>0.5) {
			QString carte16B =  QString("Geo%1%2/Z_Num%3_DeZoom%4_Geom-Im-%5.tif").arg(paramMain.getParamMicmac().at(i).getRepere()? QString("I") : QString("Ter")).arg(numRef).arg(j).arg(dezoom).arg(numRef);

			//on vérifie que ce fichier existe toujours
			if(!QFile(paramMain.getDossier() + carte16B).exists()) {
				qMessageBox(this, tr("Read error"), conv(tr("Depth map %1 not found.")).arg(carte16B));
				j++;
				dezoom /= 2;
				continue;
			}		
			QString masque = QString("Masq_Geom-Im-%1_DeZoom%2.tif").arg(numRef).arg(dezoom);
			cartes.push_back(ParamConvert8B::carte8B(carte16B, dezoom, numRef, j, masque, paramMain.convertTifName2Couleur(refImg), imgTexture));
			j++;
			if (j!=7) dezoom /= 2;	//num7 dézoom1
		}
	}

	//lancement de la fenêtre des paramètres
	paramMain.setAvancement(0);
	if (interfCartes8B!=0) delete interfCartes8B;
	interfCartes8B = new InterfCartes8B(paramMain.getDossier(), paramMain.getMicmacDir(), &cartes, this, assistant);
	interfCartes8B->show();
	int rep = interfCartes8B->exec();
	if (rep != QDialog::Accepted)
		return;
	ParamConvert8B* paramConvert8B = new ParamConvert8B(interfCartes8B->getParam());	//à modifier

	stdoutfilename = paramMain.getDossier()+QString("grShade_outstream");
	appliThread = new Cartes8BThread(&paramMain, stdoutfilename, paramConvert8B, infoLabel, &annulation);
	setUpCalc();
}

void Interface::orthoimage() {
//calcul des orthoimages mosaïquées
	if (interfOrtho==0) {
		interfOrtho = new InterfOrtho(this, assistant, paramMain, &paramMain.modifParamMicmac());
	}
	interfOrtho->show();
	int rep = interfOrtho->exec();
	if (rep != QDialog::Accepted)
		return;
	
	QApplication::setOverrideCursor( Qt::WaitCursor );
	QStringList indexImgRef;
	for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
		if (!paramMain.getParamMicmac().at(i).getOrthoCalculee()) continue;
		QString imgRef = paramMain.getParamMicmac().at(i).getImageDeReference();
		bool ok;
		QString num = paramMain.getNumImage(imgRef, &ok, false);
		if (!ok) {
			qMessageBox(this, tr("Read error."), conv(tr("Fail to extract image %1 number.")).arg(imgRef));
			return;
		}
		//on vérifie que l'ortho n'a pas déjà été calculée
		QString dossier = paramMain.getDossier() + QString("ORTHO%1/").arg(num);
		if (!interfOrtho->getEgaliser() && QFile(dossier+QString("Ortho-NonEg-Test-Redr.tif")).exists()) continue;
		if (interfOrtho->getEgaliser() && QFile(dossier+QString("Ortho-Eg-Test-Redr.tif")).exists()) continue;
		indexImgRef.push_back(num);
	}
	QVector<OrthoThread> orthoThread(indexImgRef.count(),OrthoThread(&paramMain, infoLabel, interfOrtho->getEgaliser()));
	for (int i=0; i<orthoThread.count(); i++)
		orthoThread[i].setNumImgRef(indexImgRef.at(i));
	for (int i=0; i<orthoThread.count(); i++)
		orthoThread[i].start();
	for (int i=0; i<orthoThread.count(); i++)
		while (orthoThread.at(i).isRunning()) {}
	for (int i=0; i<orthoThread.count(); i++) {
		if (!orthoThread.at(i).getIsDone()) {
			QApplication::restoreOverrideCursor();		
			qMessageBox(this, tr("Error"),orthoThread.at(i).getReaderror());
			return;
		}
	}
	if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
	if (interfModele3D!=0) interfModele3D->chercheOrtho();	//mise à jour des paramètres
}

void Interface::modeles3D()
{
	QVector<ParamNuages> nuages(7*paramMain.getParamMicmac().count(), ParamNuages());
	if (interfModele3D==0) {
		//liste des fichiers
		for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
			int dezoom = 1;
			QString carte = paramMain.getParamMicmac().at(i).getImageDeReference();
			bool ok;
			QString numCarte = paramMain.getNumImage( carte, &ok, false );
			if (!ok) {
				qMessageBox(this, tr("Read error."), conv(tr("Fail to extract image %1 number.")).arg(carte));
				return;
			}
			QString imgTexture;
			if (paramMain.getParamMicmac().at(i).getRepere()) imgTexture = paramMain.getDossier()+paramMain.convertTifName2Couleur(carte);
			else if (paramMain.getParamMicmac().at(i).getOrthoCalculee() && QFile(paramMain.getDossier()+QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(numCarte)).exists()) imgTexture = paramMain.getDossier()+QString("ORTHO%1/Ortho-Eg-Test-Redr.tif").arg(numCarte);
			else if (paramMain.getParamMicmac().at(i).getOrthoCalculee() && QFile(paramMain.getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numCarte)).exists()) imgTexture = paramMain.getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numCarte);
			else  imgTexture = paramMain.getParamMicmac().at(i).getImageSaisie(paramMain);	//TA n&b
			for (int j=7; j>0; j--) {
				nuages[7*i+j-1].setParamMasque(&paramMain.getParamMicmac().at(i));
				nuages[7*i+j-1].setCarte(imgTexture);
				nuages[7*i+j-1].setEtape(j);
				nuages[7*i+j-1].setDezoom(dezoom);
				nuages[7*i+j-1].setNumCarte(numCarte);
				nuages[7*i+j-1].calcFileName(paramMain.getDossier());
				if (j!=7) dezoom *= 2;
			}
		}
		interfModele3D = new InterfModele3D(this, assistant, paramMain, nuages);
	}
	//sélection des paramètres
	interfModele3D->show();
	int rep = interfModele3D->exec();
	if (rep != QDialog::Accepted)
		return;
	QApplication::setOverrideCursor( Qt::WaitCursor );
	nuages = interfModele3D->getModifications();

	int N = interfModele3D->getParamPly().getNuages().count(true);
	QVector<Convert2PlyThread> convert2PlyThread(N,Convert2PlyThread(&paramMain, infoLabel, interfModele3D->getParamPly()));
	int n = 0;
	for (int i=0; i<nuages.count(); i++) {
		if (!interfModele3D->getParamPly().getNuages().at(i)) continue;
		convert2PlyThread[n].setParamNuage(&(nuages.at(i)));
		n++;
	}
	for (int i=0; i<N; i++)
		convert2PlyThread[i].start();
	for (int i=0; i<N; i++)
		while (convert2PlyThread[i].isRunning()) {}
	for (int i=0; i<N; i++) {
		if (!convert2PlyThread[i].getIsDone()) {
			QApplication::restoreOverrideCursor();		
			qMessageBox(this, tr("Error"),convert2PlyThread[i].getReaderror());
			return;
		}
	}
	if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
}

void Interface::help()
{
    assistant->showDocumentation(assistant->pageInterface); 
}

void Interface::about()
{
	bool fr = (settings->value("langue").toString()==QLocale::languageToString(QLocale::French));
	QFile aboutFile((fr)? QString("../interface/help/about.txt") : QString("../interface/help/about_english.txt"));
	if (!aboutFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to find file help/about.txt.")));
		return;
	}
	qMessageBox(this, tr("About"),QTextStream(&aboutFile).readAll());
	aboutFile.close();
}

void Interface::options()
{
	//sélection des paramètres
	if (interfOptions==0) interfOptions = new InterfOptions(this, assistant, *settings);
	interfOptions->show();
	int rep = interfOptions->exec();
	if (rep != QDialog::Accepted)
		return;
	if (!interfOptions->getOptions().updateSettings(*settings))
		qMessageBox(this, tr("Read error"), conv(tr("Fail to fill up camera data base.")));
	maxcpu = interfOptions->getOptions().getCpu();
	paramMain.setMicmacDir(interfOptions->getOptions().getMicmacDir());
	paramMain.setFrench(settings->value("langue").toString()==QLocale::languageToString(QLocale::French));
}

void Interface::verifMicmac()
{
	//sélection des paramètres
	if (interfVerifMicmac==0) interfVerifMicmac = new InterfVerifMicmac(this, assistant, &paramMain, progress);
	interfVerifMicmac->show();
	interfVerifMicmac->exec();
}

void Interface::nettoyage()
{
	int rep = dispMsgBox(conv(tr("Folder clean up")), conv(tr("Do you want to delete all useless files ?")), QVector<int>()<<0<<-1<<1, 2);
	if (rep!=0) return;
	Nettoyeur nettoyeur(&paramMain);
	nettoyeur.nettoie();
}

void Interface::supprImg()
{
	if (imagesList->selectedItems().size()==0)
		return;
	
	if (imagesList->topLevelItem(0)->isSelected()) {
		paramMain.modifCorrespImgCalib().clear();
		imagesList->clear();
		return;
	}

	int i = 0;
	while(i<imagesList->topLevelItem(0)->childCount()) {
		if (imagesList->topLevelItem(0)->child(i)->isSelected()) {
			paramMain.modifCorrespImgCalib().remove(i);
			imagesList->topLevelItem(0)->removeChild(imagesList->topLevelItem(0)->child(i));
		} else
			i++;
	}

	if (paramMain.getCorrespImgCalib().size()==0) {
		updateInterface(ParamMain::BeginMode);		
		return;
	}
	saved=false;
}

void Interface::createActions()
{
	openCalcAct = new QAction(tr("&Open a project file"), this);
	openCalcAct->setShortcuts(QKeySequence::Open);
	openCalcAct->setStatusTip(tr("Load parameters of a former project"));
	connect(openCalcAct, SIGNAL(triggered()), this, SLOT(openCalc()));

	openImgAct = new QAction(tr("&Load images"), this);
	openImgAct->setShortcuts(QKeySequence::Open);
	openImgAct->setStatusTip(tr("Load images"));
	connect(openImgAct, SIGNAL(triggered()), this, SLOT(openImg()));

	saveCalcAct = new QAction(tr("&Save"), this);
	saveCalcAct->setShortcuts(QKeySequence::Save);
	saveCalcAct->setStatusTip(tr("Save current project"));
	connect(saveCalcAct, SIGNAL(triggered()), this, SLOT(saveCalc()));

	saveCalcAsAct = new QAction(tr("&Save as"), this);
	saveCalcAsAct->setShortcuts(QKeySequence::SaveAs);
	saveCalcAsAct->setStatusTip(tr("Save current project"));
	connect(saveCalcAsAct, SIGNAL(triggered()), this, SLOT(saveCalcAs()));

	exitAct = new QAction(tr("&Quit"), this);
        exitAct->setShortcut(QKeySequence::Close);
	exitAct->setStatusTip(tr("Quit application"));
        exitAct->setMenuRole(QAction::TextHeuristicRole);
	connect(exitAct, SIGNAL(triggered()), this, SLOT(closeAppli()));

	calcPastisAct = new QAction(tr("&Tie-point search"), this);
	calcPastisAct->setStatusTip(tr("Tie-point computation"));
	connect(calcPastisAct, SIGNAL(triggered()), this, SLOT(calcPastis()));

	calcAperoAct = new QAction(tr("&Pose estimation"), this);
	calcAperoAct->setStatusTip(conv(tr("Compute camera orientation")));
	connect(calcAperoAct, SIGNAL(triggered()), this, SLOT(calcApero()));

	calcMicmacAct = new QAction(tr("&Depth map computing"), this);
	calcMicmacAct->setStatusTip(tr("Compute depth maps"));
	connect(calcMicmacAct, SIGNAL(triggered()), this, SLOT(calcMicmac()));

	continueCalcAct = new QAction(tr("&Restart"), this);
	continueCalcAct->setStatusTip(tr("Restart last computation"));
	connect(continueCalcAct, SIGNAL(triggered()), this, SLOT(continueCalc()));

	vueHomolAct = new QAction(tr("&Tie-point view"), this);
	vueHomolAct->setStatusTip(tr("Display tie-points"));
	connect(vueHomolAct, SIGNAL(triggered()), this, SLOT(vueHomol()));

	vueAct = new QAction(tr("&Survey view"), this);
	vueAct->setStatusTip(tr("3D survey view"));
	connect(vueAct, SIGNAL(triggered()), this, SLOT(vue()));

	vueNuageAct = new QAction(tr("&Point cloud view"), this);
	vueNuageAct->setStatusTip(tr("Display point clouds extracted from depth maps"));
	connect(vueNuageAct, SIGNAL(triggered()), this, SLOT(vueNuages()));

	prof8BAct = new QAction(tr("&8 bit depth maps"), this);
	prof8BAct->setStatusTip(tr("Convert 16 bit depth maps into 8 bit images"));
	connect(prof8BAct, SIGNAL(triggered()), this, SLOT(cartesProf8B()));

	orthoAct = new QAction(conv(tr("&OrthoImage")), this);
	orthoAct->setStatusTip(conv(tr("Compute orthoimage mosaic")));
	connect(orthoAct, SIGNAL(triggered()), this, SLOT(orthoimage()));

	mod3DAct = new QAction(conv(tr("&3D models")), this);
	mod3DAct->setStatusTip(conv(tr("Convert depth maps into 3D models")));
	connect(mod3DAct, SIGNAL(triggered()), this, SLOT(modeles3D()));

	helpAct = new QAction(tr("&Help"), this);
	helpAct->setShortcuts(QKeySequence::HelpContents);
	helpAct->setStatusTip(tr("Help"));
	connect(helpAct, SIGNAL(triggered()), this, SLOT(help()));

	aboutAct = new QAction(tr("&About"), this);
	aboutAct->setStatusTip(tr("About"));
	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

	optionAct = new QAction(tr("&Options"), this);
	optionAct->setStatusTip(conv(tr("GUI settings")));
        optionAct->setMenuRole(QAction::TextHeuristicRole);
	connect(optionAct, SIGNAL(triggered()), this, SLOT(options()));

	verifMicmacAct = new QAction(conv(tr("&Computing checkout")),this);
	verifMicmacAct->setStatusTip(conv(tr("Check out depth map computing convergence")));
	connect(verifMicmacAct, SIGNAL(triggered()), this, SLOT(verifMicmac()));

	nettoyageAct = new QAction(conv(tr("&Folder clean up")),this);
	nettoyageAct->setStatusTip(conv(tr("Delete all useless files")));
	connect(nettoyageAct, SIGNAL(triggered()), this, SLOT(nettoyage()));

	supprImgAct = new QAction(tr("&Remove"), this);
	supprImgAct->setShortcuts(QKeySequence::Cut);
	supprImgAct->setStatusTip(tr("Remove this image from computation"));
	connect(supprImgAct, SIGNAL(triggered()), this, SLOT(supprImg()));
}

void Interface::createMenus()
{
	fileMenu = menuBarre->addMenu(tr("&File"));
	fileMenu->addAction(openCalcAct);
	fileMenu->addAction(openImgAct);
	fileMenu->addSeparator();
	fileMenu->addAction(saveCalcAct);
	fileMenu->addAction(saveCalcAsAct);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);

	calcMenu = menuBarre->addMenu(tr("&Compute"));
	calcMenu->addAction(calcPastisAct);
	calcMenu->addAction(calcAperoAct);
	calcMenu->addAction(calcMicmacAct);
	calcMenu->addSeparator();
	calcMenu->addAction(continueCalcAct);

	visuMenu = menuBarre->addMenu(tr("&View"));
	visuMenu->addAction(vueHomolAct);
	visuMenu->addAction(vueAct);
	visuMenu->addAction(vueNuageAct);

	convertMenu = menuBarre->addMenu(tr("&Conversion"));
	convertMenu->addAction(prof8BAct);
	convertMenu->addAction(orthoAct);
	convertMenu->addAction(mod3DAct);

	helpMenu = menuBarre->addMenu(tr("&Help"));
	helpMenu->addAction(helpAct);
	helpMenu->addAction(aboutAct);
	helpMenu->addAction(optionAct);
	helpMenu->addAction(verifMicmacAct);
	helpMenu->addAction(nettoyageAct);
}

void Interface::updateInterface(ParamMain::Mode mode)
{
	paramMain.setCurrentMode(mode);
	(*continueCalcAct).setEnabled(false);
	(*openCalcAct).setEnabled(true);
	(*openImgAct).setEnabled(true);
	(*exitAct).setEnabled(true);
	(*aboutAct).setEnabled(true);
	(*optionAct).setEnabled(true);

	switch (paramMain.getCurrentMode()) {
		case ParamMain::BeginMode:	//pas d'images
			saved=true;
			(*saveCalcAct).setEnabled(false);
			(*saveCalcAsAct).setEnabled(false);
			(*calcMenu).setEnabled(false);
                        (*calcPastisAct).setEnabled(false);
                        (*calcAperoAct).setEnabled(false);
                        (*calcMicmacAct).setEnabled(false);
			(*supprImgAct).setEnabled(false);
			(*visuMenu).setEnabled(false);
                        (*vueHomolAct).setEnabled(false);
                        (*vueAct).setEnabled(false);
                        (*vueNuageAct).setEnabled(false);
			(*convertMenu).setEnabled(false);
                        (*prof8BAct).setEnabled(false);
                        (*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(false);
			paramMain.setDossier(QString());
			paramMain.modifCorrespImgCalib().clear();
			imagesList->clear();
			imagesList->adjustSize ();
			pastisLabel->clear();
			aperoLabel->clear();
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);	//a priori c'est un changement de chantier ou un import d'images, paramMain est réinitialisé
			break;

		case ParamMain::ImageMode:	//les images sont importées
                       (*supprImgAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
                        (*calcPastisAct).setEnabled(true);
			(*calcAperoAct).setEnabled(false);
			(*calcMicmacAct).setEnabled(false);
			(*continueCalcAct).setEnabled(false);
			(*convertMenu).setEnabled(false);
                        (*prof8BAct).setEnabled(false);
                        (*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*visuMenu).setEnabled(false);
                        (*vueHomolAct).setEnabled(false);
                        (*vueAct).setEnabled(false);
                        (*vueNuageAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(false);
			pastisLabel->clear();
			aperoLabel->clear();
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			if (vueChantier!=0) {
				delete vueChantier;
				vueChantier = 0;
			}
			if (interfModele3D!=0) {
				delete interfModele3D;
				interfModele3D = 0;
			}
			if (interfOrtho!=0) {
				delete interfOrtho;
				interfOrtho = 0;
			}
			if (interfVerifMicmac!=0) {
				delete interfVerifMicmac;
				interfVerifMicmac = 0;
			}
			if (interfCartes8B!=0) {
				delete interfCartes8B;
				interfCartes8B = 0;
			}
			if (vueCartes!=0) {
				delete vueCartes;
				vueCartes = 0;
			}
			if (vueHomologues!=0) {	//à cause du filtrage des points
				delete vueHomologues;
				vueHomologues = 0;
			}
			if (interfMicmac!=0) {	//les images à orienter ne sont plus les mêmes
				delete interfMicmac;
				interfMicmac = 0;
			}
			if (interfApero!=0) {	//modification des images
				delete interfApero;
				interfApero = 0;
			}
			if (interfPastis!=0) {	//modification des images
				delete interfPastis;
				interfPastis = 0;
			}
			 break;

		case ParamMain::PointsEnCours:	//calcul non fini _ calcul pastis
			(*calcAperoAct).setEnabled(false);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(false);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(true);
			(*convertMenu).setEnabled(false);
                        (*prof8BAct).setEnabled(false);
                        (*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*visuMenu).setEnabled(false);
                        (*vueHomolAct).setEnabled(false);
                        (*vueAct).setEnabled(false);
                        (*vueNuageAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(false);
			pastisLabel->setText(paramMain.getTradModeInternational().at(2).second);
			aperoLabel->clear();
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			if (vueChantier!=0) {
				delete vueChantier;
				vueChantier = 0;
			}
			if (interfModele3D!=0) {
				delete interfModele3D;
				interfModele3D = 0;
			}
			if (interfOrtho!=0) {
				delete interfOrtho;
				interfOrtho = 0;
			}
			if (interfVerifMicmac!=0) {
				delete interfVerifMicmac;
				interfVerifMicmac = 0;
			}
			if (interfCartes8B!=0) {
				delete interfCartes8B;
				interfCartes8B = 0;
			}
			if (vueCartes!=0) {
				delete vueCartes;
				vueCartes = 0;
			}
			if (vueHomologues!=0) {	//à cause du filtrage des points
				delete vueHomologues;
				vueHomologues = 0;
			}
			if (interfMicmac!=0) {	//les images à orienter ne sont plus les mêmes
				delete interfMicmac;
				interfMicmac = 0;
			}
			if (interfApero!=0) {	//il peut y avoir des changments (calibrations ?)
				delete interfApero;
				interfApero = 0;
			}
			break;

		case ParamMain::PointsMode:	//points de liaison trouvés
			(*calcAperoAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(false);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(false);
			(*visuMenu).setEnabled(true);
			(*vueHomolAct).setEnabled(true);
			(*vueAct).setEnabled(false);
			(*vueNuageAct).setEnabled(false);
			(*convertMenu).setEnabled(false);
			(*prof8BAct).setEnabled(false);
			(*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(true);
			pastisLabel->setText(paramMain.getTradModeInternational().at(3).second);
			aperoLabel->clear();
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			break;

		case ParamMain::PoseEnCours:	//calcul non fini _ calcul apero
			(*calcAperoAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(false);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(true);
			(*visuMenu).setEnabled(true);
			(*vueHomolAct).setEnabled(true);
			(*vueAct).setEnabled(false);
			(*vueNuageAct).setEnabled(false);
			(*convertMenu).setEnabled(false);
			(*prof8BAct).setEnabled(false);
			(*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(false);
			pastisLabel->setText(paramMain.getTradModeInternational().at(3).second);
			aperoLabel->setText(paramMain.getTradModeInternational().at(4).second);
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			if (vueChantier!=0) {
				delete vueChantier;
				vueChantier = 0;
			}
			if (interfModele3D!=0) {
				delete interfModele3D;
				interfModele3D = 0;
			}
			if (interfOrtho!=0) {
				delete interfOrtho;
				interfOrtho = 0;
			}
			if (interfVerifMicmac!=0) {
				delete interfVerifMicmac;
				interfVerifMicmac = 0;
			}
			if (interfCartes8B!=0) {
				delete interfCartes8B;
				interfCartes8B = 0;
			}
			if (vueCartes!=0) {
				delete vueCartes;
				vueCartes = 0;
			}
			if (vueHomologues!=0) {	//à cause du filtrage des points
				delete vueHomologues;
				vueHomologues = 0;
			}
			if (interfMicmac!=0) {	//les images à orienter ne sont plus les mêmes
				delete interfMicmac;
				interfMicmac = 0;
			}
			break;

		case ParamMain::PoseMode:	//poses calculées
			(*calcAperoAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(true);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(false);
			(*visuMenu).setEnabled(true);
			(*vueHomolAct).setEnabled(true);
			(*vueAct).setEnabled(true);
			(*vueNuageAct).setEnabled(false);
			(*convertMenu).setEnabled(false);
			(*prof8BAct).setEnabled(false);
			(*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(true);
			pastisLabel->setText(paramMain.getTradModeInternational().at(3).second);
			aperoLabel->setText(paramMain.getTradModeInternational().at(5).second);
			micmacLabel->clear();
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			break;

		case ParamMain::CarteEnCours:	//calcul non fini _ calcul micmac
			(*calcAperoAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(true);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(true);
			(*visuMenu).setEnabled(true);
			(*vueHomolAct).setEnabled(true);
			(*vueAct).setEnabled(true);
			(*vueNuageAct).setEnabled(false);
                        (*convertMenu).setEnabled(false);
			(*prof8BAct).setEnabled(false);
			(*mod3DAct).setEnabled(false);
                        (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(false);
			pastisLabel->setText(paramMain.getTradModeInternational().at(3).second);
			aperoLabel->setText(paramMain.getTradModeInternational().at(5).second);
			micmacLabel->setText(paramMain.getTradModeInternational().at(6).second);
			infoLabel->clear();
			infoLabel->setFrameStyle(QFrame::NoFrame);
			if (interfModele3D!=0) {
				delete interfModele3D;
				interfModele3D = 0;
			}
			if (interfOrtho!=0) {
				delete interfOrtho;
				interfOrtho = 0;
			}
			if (interfVerifMicmac!=0) {
				delete interfVerifMicmac;
				interfVerifMicmac = 0;
			}
			if (interfCartes8B!=0) {
				delete interfCartes8B;
				interfCartes8B = 0;
			}
			if (vueCartes!=0) {
				delete vueCartes;
				vueCartes = 0;
			}
			break;

		case ParamMain::EndMode:	//poses calculées
			(*calcAperoAct).setEnabled(true);
			(*saveCalcAct).setEnabled(true);
			(*saveCalcAsAct).setEnabled(true);
			(*calcMenu).setEnabled(true);
			(*calcPastisAct).setEnabled(true);
			(*calcMicmacAct).setEnabled(true);
			(*supprImgAct).setEnabled(false);
			(*continueCalcAct).setEnabled(false);
			(*visuMenu).setEnabled(true);
			(*vueHomolAct).setEnabled(true);
			(*vueAct).setEnabled(true);
			(*vueNuageAct).setEnabled(true);
			(*convertMenu).setEnabled(true);
			(*prof8BAct).setEnabled(true);
			(*mod3DAct).setEnabled(true);
			bool b = false;
			for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
				if (!paramMain.getParamMicmac().at(i).getRepere() && paramMain.getParamMicmac().at(i).getDoOrtho()) {
					b = true;
					break;
				}
			}
                        if (b) (*orthoAct).setEnabled(true);
			else (*orthoAct).setEnabled(false);
			(*verifMicmacAct).setEnabled(false);
			(*nettoyageAct).setEnabled(true);
			pastisLabel->setText(paramMain.getTradModeInternational().at(3).second);
			aperoLabel->setText(paramMain.getTradModeInternational().at(5).second);
			micmacLabel->setText(paramMain.getTradModeInternational().at(7).second);
			if (!infoLabel->text().isEmpty()) {
				infoLabel->setFrameStyle(QFrame::Box);
				infoLabel->adjustSize();
			} else
				infoLabel->setFrameStyle(QFrame::NoFrame);
			break;
	}
	if (paramMain.getCurrentMode()!=ParamMain::BeginMode && paramMain.getCurrentMode()!=ParamMain::ImageMode) { 
		int i = 0;
		while (paramMain.getParamPastis().getTradTypChan().at(i).first != paramMain.getParamPastis().getTypeChantier() && i<paramMain.getParamPastis().getTradTypChan().count()) i++;
		chantierLabel->setText(QString("	->   ")+paramMain.getParamPastis().getTradTypChanInternational().at(i).second);
	} else 
		chantierLabel->clear();
 }

ParamMain& Interface::modifParamMain() { return paramMain; }

void Interface::setStdOutFile(const QString& fichier) { stdoutfilename = fichier; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//chronomètre
Timer::Timer(QString dossier) : QTime() {
	dir = dossier;
}
void Timer::displayTemps(QString label) {
	//calcul du temps écoulé
	double tps = double(restart())/1000.0;
	int minutes = int(floor(tps))/60;
	double secondes = tps - 60 * minutes;
	QString ligne = label + QVariant(tps).toString() + QString(" s (") + QVariant(minutes).toString() + QString(" min ") + QVariant(secondes).toString() + QString(" s)");
	//écriture
	QFile file(dir+QString("timer"));
	if (!file.open(QIODevice::WriteOnly | QIODevice::Append)) {
		cout << QObject::tr("Problem in writing file timer").toStdString() << endl;
		return;
	}
	QTextStream outStream(&file);
	outStream << ligne.toStdString().c_str() << "\n";
	file.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FileDialog::FileDialog(QWidget* parent, const QString& caption, const QString& directory, const QString& filter) : QFileDialog(parent, caption, directory, filter) {
	setLabelText(QFileDialog::LookIn, conv(tr("Look in")));
	setLabelText(QFileDialog::FileName, conv(tr("File name :")));
	setLabelText(QFileDialog::FileType, conv(tr("File type :")));
	setLabelText(QFileDialog::Accept, conv(tr("Accept")));
	setLabelText(QFileDialog::Reject, conv(tr("Cancel")));
}
FileDialog::~FileDialog() {}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//paramètres des images
pair<int,int> ParamImage::numPos(-1,-1);
ParamImage::ParamImage() : imageRAW(QString()), imageTif(QString()), calibration(QString()), taille(QSize(0,0)), numero(QString()) {}
ParamImage::ParamImage(const ParamImage& paramImage) { copie(paramImage); }

ParamImage& ParamImage::operator=(const ParamImage& paramImage) {
	if (&paramImage!=this)
		copie(paramImage);
        return *this;
}
void ParamImage::copie(const ParamImage& paramImage) {
	imageRAW = paramImage.getImageRAW();
	imageTif = paramImage.getImageTif();
	calibration = paramImage.getCalibration();
	taille = paramImage.getTaille();
	numero = paramImage.getNumero();
}

bool ParamImage::isEqualTo(const QString& image, int type) const {
	switch (type) {
		case 0 : return (image == imageRAW);
		case 2 : return (image == imageTif);
		default : return false;
	}
}
bool ParamImage::isEqualTo(const ParamImage& paramImage, int type) const {
	switch (type) {
		case 0 : return (paramImage.getImageRAW() == imageRAW);
		case 2 : return (paramImage.getImageTif() == imageTif);
		default : return false;
	}
}

const QString& ParamImage::getImageRAW() const {return imageRAW; }
const QString& ParamImage::getImageTif() const {return imageTif; }
const QString& ParamImage::getCalibration() const {return calibration; }
const QSize& ParamImage::getTaille() const {return taille; }
QString ParamImage::getNumero() const {return numero; }

void ParamImage::setImageRAW(const QString& raw) { imageRAW = raw; }
void ParamImage::setImageTif(const QString& tif) { imageTif = tif; }
void ParamImage::setCalibration(const QString& calib) { calibration = calib; }
void ParamImage::setTaille(const QSize& size) { taille = size; }
bool ParamImage::calcNumero(const ParamMain& paramMain) {
	bool ok;
	numero = paramMain.getNumImage(imageTif,&ok,false);
	return ok;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//paramètres du calcul
QStringList ParamMain::formatsImg = QStringList("cr2")<<"3fr"<<"arw"<<"crw"<<"dng"<<"kdc"<<"mrw"<<"nef"<<"orf"<<"pef"<<"ptx"<<"raf"<<"x3f"<<"rw2"<<"jpg";
QString ParamMain::imageXML("Liste_images.xml");
QString ParamMain::coupleXML("Liste_couples.xml");
QString ParamMain::assocCalibXML("Liste_calibrations_internes.xml");
QString ParamMain::postfixTifCouleur("_couleur");
QString ParamMain::chantierXML("MicMac-LocalChantierDescripteur.xml");
QString ParamMain::aperoXML("Apero.xml");
QString ParamMain::maitresseXML("Image_Maitresse.xml");
QString ParamMain::exportPlyXML("ExportPly.xml");
QString ParamMain::cleCalibFishEyeXML("CleCalibFishEye.xml");
QString ParamMain::cleCalibClassiqXML("CleCalibClassiq.xml");
QString ParamMain::contraintesXML("Contraintes.xml");
QString ParamMain::imgOriAutoCalibXML("Images_a_orienter_autocalibration.xml");
QString ParamMain::imgOriXML("Images_a_orienter.xml");
QString ParamMain::calibDefXML("DefCalibration.xml");
QString ParamMain::cleCalibCourtXML("CleCalibCourtes.xml");
QString ParamMain::cleCalibCourtClassiqXML("CleCalibCourtClassiq.xml");
QString ParamMain::cleCalibCourtFishEyeXML("CleCalibCourtFishEye.xml");
QString ParamMain::cleCalibLongClassiqXML("CleCalibLonguesClassiq.xml");
QString ParamMain::cleCalibLongFishEyeXML("CleCalibLonguesFishEye.xml");
QString ParamMain::defCalibVInitXML("DefCalibrationAcVInit.xml");
QString ParamMain::defCalibCourtXML("DefCalibrationCourte.xml");
QString ParamMain::imgsOriVInitXML("Images_a_orienter_acVInit.xml");
QString ParamMain::imgsCourtOriXML("Images_courtes_a_orienter.xml");
QString ParamMain::posesFigeesXML("PosesFigees.xml");
QString ParamMain::posesLibresXML("PosesLibres.xml");
QString ParamMain::defCalibTtInitXML("DefCalibrationToutInit.xml");
QString ParamMain::imgsOriTtInitXML("Images_a_orienter_ToutInit.xml");
QString ParamMain::posesNonDissocXML("PosesNonDissoc.xml");
QString ParamMain::cleCalibLiberFishEyeXML("CleCalibLiberFishEye.xml");
QString ParamMain::cleCalibLiberClassiqXML("CleCalibLiberClassiq.xml");
QString ParamMain::orientationAbsolueXML("OrientationAbsolue.xml");
QString ParamMain::orientationGPSXML("OrientationGPS.xml");
QString ParamMain::defObsGPSXML("DefObservationsGPS.xml");
QString ParamMain::defIncGPSXML("DefInconnuesGPS.xml");
QString ParamMain::ponderationGPSXML("PonderationsGPS.xml");
QString ParamMain::oriInitXML("OriInit.xml");
QString ParamMain::micmacXML("param-GeoIm.xml");
QString ParamMain::micmacTerXML("param-Terrain.xml");
QString ParamMain::micmacMMOrthoXML("param-Ortho.xml");
QString ParamMain::intervalleXML("Intervalle.xml");
QString ParamMain::discontinuteXML("Discontinuites.xml");
QString ParamMain::defMasqueXML("DefMasque.xml");
QString ParamMain::cartesXML("Cartes_A_Creer.xml");
QString ParamMain::repereXML("Repere.xml");
QString ParamMain::nomCartesXML("Nom_carte.xml");
QString ParamMain::nomTAXML("Nom_TA.xml");
QString ParamMain::orthoXML("Ortho.xml");
QString ParamMain::paramPortoXML("param-Porto.xml");
QString ParamMain::portoXML("Porto.xml");
QString ParamMain::pathMntXML("PathMNT.xml");

ParamMain::ParamMain():
	french( true ),
	tradMode( QVector<pair<Mode,QString> >(8) ),
	tradModeInternational( QVector<pair<Mode,QString> >(8) ),
	micmacDir( QString() ),
	avancement( 0 ),
	correspImgCalib( 0 ),
	paramPastis( 0 ),
	paramApero( 0 ),
	paramMicmac( QVector<CarteDeProfondeur>() )
{
	tradMode[0] = pair<Mode,QString>(BeginMode, conv("initialisation")) ;
	tradMode[1] = pair<Mode,QString>(ImageMode, conv("images importees")) ;
	tradMode[2] = pair<Mode,QString>(PointsEnCours, conv("calcul des points homologues en cours")) ;
	tradMode[3] = pair<Mode,QString>(PointsMode, conv("points homologues calcules")) ;
	tradMode[4] = pair<Mode,QString>(PoseEnCours, conv("calcul des poses en cours")) ;
	tradMode[5] = pair<Mode,QString>(PoseMode, conv("poses calculees")) ;
	tradMode[6] = pair<Mode,QString>(CarteEnCours, conv("calcul des cartes de profondeur en cours")) ;
	tradMode[7] = pair<Mode,QString>(EndMode, conv("cartes de profondeur calculees")) ;

	tradModeInternational[0] = pair<Mode,QString>(BeginMode, conv(QObject::tr("Initialisation"))) ;
	tradModeInternational[1] = pair<Mode,QString>(ImageMode, conv(QObject::tr("Loaded images"))) ;
	tradModeInternational[2] = pair<Mode,QString>(PointsEnCours, conv(QObject::tr("Tie-point compution in progress"))) ;
	tradModeInternational[3] = pair<Mode,QString>(PointsMode, conv(QObject::tr("Tie-point search done"))) ;
	tradModeInternational[4] = pair<Mode,QString>(PoseEnCours, conv(QObject::tr("Pose compution in progress"))) ;
	tradModeInternational[5] = pair<Mode,QString>(PoseMode, conv(QObject::tr("Pose estimation done"))) ;
	tradModeInternational[6] = pair<Mode,QString>(CarteEnCours, conv(QObject::tr("Depth map compution in progress"))) ;
	tradModeInternational[7] = pair<Mode,QString>(EndMode, conv(QObject::tr("Depth map computation done"))) ;

	init();
}
ParamMain::ParamMain(const ParamMain& paramMain) { copie(this,paramMain); }
ParamMain::~ParamMain() {
	delete paramPastis;
	delete paramApero;
}

ParamMain& ParamMain::operator=(const ParamMain& paramMain) {
	if (&paramMain!=this)
		copie(this,paramMain);
	return *this;
}

void copie(ParamMain* paramMain1, const ParamMain& paramMain2) {
	paramMain1->tradMode = paramMain2.tradMode;
	paramMain1->tradModeInternational = paramMain2.tradModeInternational;
	paramMain1->micmacDir = paramMain2.micmacDir;
	paramMain1->calculXML = paramMain2.calculXML;
	paramMain1->currentMode = paramMain2.currentMode;
	paramMain1->avancement = paramMain2.avancement;
	paramMain1->etape = paramMain2.etape;
	paramMain1->dossier = paramMain2.dossier;
	paramMain1->correspImgCalib = paramMain2.correspImgCalib;
	paramMain1->paramPastis = paramMain2.paramPastis;
	paramMain1->makeFile = paramMain2.makeFile;
	paramMain1->paramApero = paramMain2.paramApero;
	paramMain1->paramMicmac = paramMain2.paramMicmac;
}

void ParamMain::init() {	
	setDossier(QString());
	setCalculXML(QString());
	setAvancement(0);
	setCurrentMode(ParamMain::BeginMode);
	setMakeFile(QString());
	correspImgCalib = QVector<ParamImage>();
	 if (paramPastis!=0) delete paramPastis;
	paramPastis = new ParamPastis();
	if (paramApero!=0) delete paramApero;
	paramApero = new ParamApero;
	paramMicmac.clear();
}

bool ParamMain::isValideFormat(const QString& extension) {
//tous les formats raw et le jpg
	bool valide = false;
	for (int j=0; j<formatsImg.count(); j++) {
		if (extension.toUpper()==formatsImg.at(j).toUpper()) {
			valide = true;
			break;
		}
	}
	return valide;
}

QString ParamMain::getNumImage(const QString& image, bool* ok, bool TA) const {
//extrait le numéro d'une image du type : préfixe + n° + postfixe + _.tif ; si TA=true, l'image est un TA de type TA + n° + /TA_Geom-Im- + n° + .tif ou TA + n° + /TA_LeChantier.tif
	if (!TA) {
		if (ParamImage::numPos.first==-1 || ParamImage::numPos.second==-1) {
			if (ok!=0) *ok = false;
			return QString();
		}
		QString image1 = image.section("/",-1,-1).section(".",0,-2);	
		image1 = image1.left(image1.count()-ParamImage::numPos.second);
		QString res = image1.right(image1.count()-ParamImage::numPos.first);
		//res.toInt(ok);
		//if (!*ok) return QString();	//pas forcément un nombre si les noms des images ont une décomposition différente
		if (ok!=0) *ok = true;
		return res;
	} else {
		QString image2 = image.section("/", -2, -2);
		image2 = image2.right( image2.size() - 2 );
		if (image.section("/",-2,-1)!=QString("TA%1/TA_Geom-Im-%1.tif").arg(image2) && image.section("/",-2,-1)!=QString("TA%1/TA_LeChantier.tif").arg(image2)) {
			if (ok!=0) *ok = false;
			return QString();
		}
		//image2.toInt(ok);
		//if (!*ok) return QString();
		if (ok!=0) *ok = true;
		return image2;
	}
}

bool ParamMain::calcImgsId() {
	//recherche de la chaîne minimale pour désigner les images (plus simple pour l'utilisateur que de donner un numéro aléatoire)
	int pos0, posN;
	pos0 = posN = -1;
	QString img1 = correspImgCalib.at(0).getImageTif().section(".",0,-2);
	QString img2 = correspImgCalib.at(1).getImageTif().section(".",0,-2);
	for (int i=0; i<min(img1.count(),img2.count()); i++) {
		if (img1.at(i)!=img2.at(i)) {
			pos0 = i;
			break;
		}
	}
	for (int i=0; i<=img1.count()-1-pos0; i++) {
		if (img1.at(img1.count()-1-i)!=img2.at(img2.count()-1-i)) {
			posN = i;
			break;
		}
	}

	if (correspImgCalib.count()>2) {
		for (int i=2; i<correspImgCalib.count(); i++) {
			img2 = correspImgCalib.at(i).getImageTif().section(".",0,-2);
			if (pos0>0) {
				for (int j=0; j<pos0; j++) {
					if (img1.at(j)!=img2.at(j)) {
						pos0 = j;
						break;
					}
				}
			}
			if (posN>0) {
				for (int j=0; j<posN; j++) {
					if (img1.at(img1.count()-1-j)!=img2.at(img2.count()-1-j)) {
						posN = j;
						break;
					}
				}
			}
		}
	}

	ParamImage::numPos = pair<int,int>(pos0, posN);

	//calcul de l'identifiant pour chaque image
	for (int i=0; i<correspImgCalib.count(); i++)
		if (!correspImgCalib[i].calcNumero(*this)) return false;
	return true;
}

int ParamMain::findImg(const QString& image, int type, bool strict) const {
	for (int i=0; i<getCorrespImgCalib().count(); i++) {
		if (type==0 && getCorrespImgCalib().at(i).getImageRAW()==image)
			return i;
		else if (type==0 && !strict && getCorrespImgCalib().at(i).getImageRAW().right(getCorrespImgCalib().at(i).getImageRAW().count()-5)==image)
			return i;
		else if (type==1 && getCorrespImgCalib().at(i).getImageTif()==image)
			return i;
		else if (type==1 && !strict && getCorrespImgCalib().at(i).getImageTif().right(getCorrespImgCalib().at(i).getImageTif().count()-5)==image)
			return i;
	}
	return -1;
}

QString ParamMain::saveImgsSize() {	//à n'utiliser que après PastisThread::Ecriture et PtsInteret si ouverture d'un calcul
	//récupération des tailles des images et enregistrement dans getCorrespImgCalib
	QVector<QSize> V(getParamPastis().getCalibFiles().count(), QSize(0,0));	//en fait on ne lit la taille qu'une fois par calibration (c'est trop long de la calculer pour chaque image)
	for (int i=0; i<getCorrespImgCalib().count(); i++) {
		if (getCorrespImgCalib().at(i).getTaille()!=QSize(0,0)) {
			continue;	//déjà fait
		}	
		//calibration correspondant à l'image i
		int N = -1;
		for (int j=0; j<getParamPastis().getCalibFiles().count(); j++) {
			if (getParamPastis().getCalibFiles().at(j).first==getCorrespImgCalib().at(i).getCalibration()) {
				N = j;
				break;
		       }
		}
		if (N==-1)
			return conv(QObject::tr("Image %1 calibration file not found.")).arg(dossier+getCorrespImgCalib().at(i).getImageTif());
		if (V.at(N)==QSize(0,0)) {
			QSize taille;
			if (!focaleTif(dossier+getCorrespImgCalib().at(i).getImageTif(), micmacDir, 0, &taille))
				return conv(QObject::tr("Fail to read image %1 size.")).arg(getCorrespImgCalib().at(i).getImageTif());
			V[N] = taille;
		}
		modifCorrespImgCalib()[i].setTaille(V.at(N));
	}
	return QString();
}

QString ParamMain::convertTifName2Couleur(const QString& image) const {
	if (image.right(postfixTifCouleur.count()+4)==postfixTifCouleur+QString(".tif")) return image;
	else return image.section(".",0,-2)+postfixTifCouleur+QString(".tif");
}

int ParamMain::nbCartesACalculer() const {
	int nb = 0;
	for (int i=0; i<paramMicmac.count(); i++) {
		if (paramMicmac.at(i).getACalculer()) nb++;
	}
	return nb;
}

bool ParamMain::isFrench() const {return french;}
const QStringList& ParamMain::getFormatsImg() const {return formatsImg;}
const QVector<pair<ParamMain::Mode,QString> >& ParamMain::getTradMode() const {return tradMode;}
const QVector<pair<ParamMain::Mode,QString> >& ParamMain::getTradModeInternational() const {return tradModeInternational;}
const QString& ParamMain::getMicmacDir() const {return micmacDir;}
const QString& ParamMain::getCalculXML() const {return calculXML;}
const ParamMain::Mode& ParamMain::getCurrentMode() const {return currentMode;}
int ParamMain::getAvancement() const {return avancement;}
int ParamMain::getEtape() const {return etape;}
const QString& ParamMain::getDossier() const {return dossier;}
const ParamPastis& ParamMain::getParamPastis() const {return *paramPastis;}
ParamPastis& ParamMain::modifParamPastis() {return *paramPastis;}
const QString& ParamMain::getImageXML() {return imageXML;}
const QString& ParamMain::getCoupleXML() {return coupleXML;}
const QString& ParamMain::getAssocCalibXML() {return assocCalibXML;}
const QVector<ParamImage>& ParamMain::getCorrespImgCalib() const {return correspImgCalib;}
QVector<ParamImage>& ParamMain::modifCorrespImgCalib() {return correspImgCalib;}
const QString& ParamMain::getChantierXML() {return chantierXML;}
const QString& ParamMain::getMakeFile() const {return makeFile;}
const ParamApero& ParamMain::getParamApero() const {return *paramApero;}
ParamApero& ParamMain::modifParamApero() {return *paramApero;}
const QString& ParamMain::getAperoXML() {return aperoXML;}
const QString& ParamMain::getMaitresseXML() {return maitresseXML;}
const QString& ParamMain::getExportPlyXML() {return exportPlyXML;}
const QString& ParamMain::getCleCalibFishEyeXML() {return cleCalibFishEyeXML;}
const QString& ParamMain::getCleCalibClassiqXML() {return cleCalibClassiqXML;}
const QString& ParamMain::getContraintesXML() {return contraintesXML;}
const QString& ParamMain::getImgOriAutoCalibXML() {return imgOriAutoCalibXML;}
const QString& ParamMain::getImgOriXML() {return imgOriXML;}
const QString& ParamMain::getCalibDefXML() {return calibDefXML;}
const QString& ParamMain::getCleCalibCourtXML() {return cleCalibCourtXML;}
const QString& ParamMain::getCleCalibCourtFishEyeXML() {return cleCalibCourtFishEyeXML;}
const QString& ParamMain::getCleCalibCourtClassiqXML() {return cleCalibCourtClassiqXML;}
const QString& ParamMain::getCleCalibLongFishEyeXML() {return cleCalibLongFishEyeXML;}
const QString& ParamMain::getCleCalibLongClassiqXML() {return cleCalibLongClassiqXML;}
const QString& ParamMain::getDefCalibVInitXML() {return defCalibVInitXML;}
const QString& ParamMain::getDefCalibCourtXML() {return defCalibCourtXML;}
const QString& ParamMain::getImgsOriVInitXML() {return imgsOriVInitXML;}
const QString& ParamMain::getImgsCourtOriXML() {return imgsCourtOriXML;}
const QString& ParamMain::getPosesFigeesXML() {return posesFigeesXML;}
const QString& ParamMain::getPosesLibresXML() {return posesLibresXML;}
const QString& ParamMain::getDefCalibTtInitXML() {return defCalibTtInitXML;}
const QString& ParamMain::getImgsOriTtInitXML() {return imgsOriTtInitXML;}
const QString& ParamMain::getPosesNonDissocXML() {return posesNonDissocXML;}
const QString& ParamMain::getCleCalibLiberFishEyeXML() {return cleCalibLiberFishEyeXML;}
const QString& ParamMain::getCleCalibLiberClassiqXML() {return cleCalibLiberClassiqXML;}
const QString& ParamMain::getOrientationGPSXML() {return orientationGPSXML;}
const QString& ParamMain::getOrientationAbsolueXML() {return orientationAbsolueXML;}
const QString& ParamMain::getDefObsGPSXML() {return defObsGPSXML;}
const QString& ParamMain::getDefIncGPSXML() {return defIncGPSXML;}
const QString& ParamMain::getPonderationGPSXML() {return ponderationGPSXML;}
const QString& ParamMain::getOriInitXML() {return oriInitXML;}
const QString& ParamMain::getMicmacXML() {return micmacXML;}
const QString& ParamMain::getMicmacTerXML() {return micmacTerXML;}
const QString& ParamMain::getMicmacMMOrthoXML() {return micmacMMOrthoXML;}
const QVector<CarteDeProfondeur>& ParamMain::getParamMicmac() const {return paramMicmac;}
QVector<CarteDeProfondeur>& ParamMain::modifParamMicmac() {return paramMicmac;}
const QString& ParamMain::getIntervalleXML() {return intervalleXML;}
const QString& ParamMain::getDiscontinuteXML() {return discontinuteXML;}
const QString& ParamMain::getDefMasqueXML() {return defMasqueXML;}
const QString& ParamMain::getCartesXML() {return cartesXML;}
const QString& ParamMain::getRepereXML() {return repereXML;}
const QString& ParamMain::getNomCartesXML() {return nomCartesXML;}
const QString& ParamMain::getNomTAXML() {return nomTAXML;}
const QString& ParamMain::getOrthoXML() {return orthoXML;}
const QString& ParamMain::getParamPortoXML() {return paramPortoXML;}
const QString& ParamMain::getPortoXML() {return portoXML;}
const QString& ParamMain::getPathMntXML() {return pathMntXML;}

void ParamMain::setFrench(bool f) {french = f;}
void ParamMain::setMicmacDir(const QString& micmacDossier) {micmacDir = micmacDossier;}
void ParamMain::setCalculXML(const QString& calculFile) {calculXML = calculFile;}
void ParamMain::setCurrentMode(Mode mode) {currentMode=mode;}
void ParamMain::setAvancement(int n) {avancement=n;}
void ParamMain::setEtape(int n) {etape=n;}
void ParamMain::setDossier(const QString& dir) {dossier=dir;}
void ParamMain::setParamPastis(const ParamPastis& pPastis) {*paramPastis=pPastis; }
void ParamMain::setCorrespImgCalib(const QVector<ParamImage>& array) {correspImgCalib=array;}
void ParamMain::setMakeFile(const QString& file) {makeFile=file;}
void ParamMain::setParamApero(const ParamApero& pApero) {*paramApero=pApero;}
void ParamMain::setParamMicmac(const QVector<CarteDeProfondeur>& p) {paramMicmac=p;}
