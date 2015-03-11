#include "assistant.h"

using namespace std;

Assistant::Assistant() : proc(0) {}
Assistant::Assistant(const Assistant& assistant) : proc(0) {}
Assistant::~Assistant() {
	if (proc && proc->state() == QProcess::Running) {
		proc->terminate();
		proc->waitForFinished(3000);
	}
	delete proc;
}

void Assistant::setPages(bool fr) {
	if (fr) {
		pageInterface = QString("documentation.html#__RefHeading__498_469008054");

		pageInterPastis = QString("documentation.html#__RefHeading__524_469008054");
		pageInterPastisChoice = QString("documentation.html#__RefHeading__528_469008054");
		pageInterPastisCouple = QString("documentation.html#__RefHeading__530_469008054");
		pageInterPastisCommun = QString("documentation.html#__RefHeading__532_469008054");

		pageInterfApero = QString("documentation.html#__RefHeading__540_469008054");
		pageInterfAperoMaitresse = QString("documentation.html#__RefHeading__544_469008054");
		pageInterfAperoReference = QString("documentation.html#__RefHeading__546_469008054");
		pageInterfAperoOrInit = QString("documentation.html#__RefHeading__548_469008054");
		pageInterfAperoAutocalib = QString("documentation.html#__RefHeading__550_469008054");
		pageInterfAperoMultiechelle = QString("documentation.html#__RefHeading__552_469008054");
		pageInterfAperoLibercalib = QString("documentation.html#__RefHeading__554_469008054");
		pageInterfAperoPtshomol = QString("documentation.html#__RefHeading__556_469008054");

		pageInterfMicmac = QString("documentation.html#__RefHeading__564_469008054");
		pageInterfMicmacMNT = QString("documentation.html#__RefHeading__568_469008054");
		pageInterfMicmacRepere = QString("documentation.html#__RefHeading__570_469008054");
		pageInterfMicmacMasque = QString("documentation.html#__RefHeading__572_469008054");
		pageInterfMicmacOrtho = QString("documentation.html#__RefHeading__574_469008054");
		pageInterfMicmacProfondeur = QString("documentation.html#__RefHeading__576_469008054");

		pageInterfCartes8B = QString("documentation.html#__RefHeading__600_469008054");
		pageInterfModeles3D = QString("documentation.html#__RefHeading__602_469008054");
		pageInterfOrtho = QString("documentation.html#__RefHeading__604_469008054");

		pageVueChantier = QString("documentation.html#__RefHeading__596_469008054");
		pageVueNuages = QString("documentation.html#__RefHeading__598_469008054");

		pageVueHomologues = QString("documentation.html#__RefHeading__594_469008054");
		pageDrawSegment = QString("documentation.html#__RefHeading__584_469008054");
		pageDrawPlanCorrel = QString("documentation.html#__RefHeading__584_469008054");

		pageInterfOptions = QString("documentation.html#__RefHeading__520_469008054");
		pageInterfVerifMicmac = QString("documentation.html#__RefHeading__580_469008054");
	} else {
		pageInterface = QString("documentation_english.html#__RefHeading__1103_927265466");

		pageInterPastis = QString("documentation_english.html#__RefHeading__1113_927265466");
		pageInterPastisChoice = QString("documentation_english.html#__RefHeading__1117_927265466");
		pageInterPastisCouple = QString("documentation_english.html#__1119_927265466");
		pageInterPastisCommun = QString("documentation_english.html#__RefHeading__1121_927265466");

		pageInterfApero = QString("documentation_english.html#__RefHeading__1129_927265466");
		pageInterfAperoMaitresse = QString("documentation_english.html#__RefHeading__1133_927265466");
		pageInterfAperoReference = QString("documentation_english.html#__RefHeading__1135_927265466");
		pageInterfAperoOrInit = QString("documentation_english.html#__RefHeading__1137_927265466");
		pageInterfAperoAutocalib = QString("documentation_english.html#__RefHeading__625_117990941");
		pageInterfAperoMultiechelle = QString("documentation_english.html#__RefHeading__627_117990941");
		pageInterfAperoLibercalib = QString("documentation_english.html#__RefHeading__629_117990941");
		pageInterfAperoPtshomol = QString("documentation_english.html#__RefHeading__631_117990941");

		pageInterfMicmac = QString("documentation_english.html#__RefHeading__1153_927265466");
		pageInterfMicmacMNT = QString("documentation_english.html#__RefHeading__581_2807389221");
		pageInterfMicmacRepere = QString("documentation_english.html#__RefHeading__1159_927265466");
		pageInterfMicmacMasque = QString("documentation_english.html#__RefHeading__1561_927265466");
		pageInterfMicmacOrtho = QString("documentation_english.html#__RefHeading__1466_1113520036");
		pageInterfMicmacProfondeur = QString("documentation_english.html#__RefHeading__1468_1113520036");

		pageInterfCartes8B = QString("documentation_english.html#__RefHeading__1189_927265466");
		pageInterfModeles3D = QString("documentation_english.html#__RefHeading__1191_927265466");
		pageInterfOrtho = QString("documentation_english.html#__RefHeading__1193_927265466");

		pageVueChantier = QString("documentation_english.html#__RefHeading__1185_927265466");
		pageVueNuages = QString("documentation_english.html#__RefHeading__1187_927265466");

		pageVueHomologues = QString("documentation_english.html#__RefHeading__1183_927265466");
		pageDrawSegment = QString("documentation_english.html#__RefHeading__1173_927265466");
		pageDrawPlanCorrel = QString("documentation_english.html#__RefHeading__1173_927265466");

		pageInterfOptions = QString("documentation_english.html#__RefHeading__1109_927265466");
		pageInterfVerifMicmac = QString("documentation_english.html#__RefHeading__1470_1113520036");
	}
}

Assistant& Assistant::operator=(const Assistant& assistant) {
	if (&assistant!=this) {
		proc = 0;
		pageInterface = assistant.pageInterface;
		pageInterPastis = assistant.pageInterPastis;
		pageInterPastisChoice = assistant.pageInterPastisChoice;
		pageInterPastisCouple = assistant.pageInterPastisCouple;
		pageInterPastisCommun = assistant.pageInterPastisCommun;
		pageInterfApero = assistant.pageInterfApero;
		pageInterfAperoMaitresse = assistant.pageInterfAperoMaitresse;
		pageInterfAperoReference = assistant.pageInterfAperoReference;
		pageInterfAperoAutocalib = assistant.pageInterfAperoAutocalib;
		pageInterfAperoMultiechelle = assistant.pageInterfAperoMultiechelle;
		pageInterfAperoLibercalib = assistant.pageInterfAperoLibercalib;
		pageInterfAperoPtshomol = assistant.pageInterfAperoPtshomol;
		pageInterfMicmac = assistant.pageInterfMicmac;
		pageInterfMicmacProfondeur = assistant.pageInterfMicmacProfondeur;
		pageInterfCartes8B = assistant.pageInterfCartes8B;
		pageInterfModeles3D = assistant.pageInterfModeles3D;
		pageInterfOrtho = assistant.pageInterfOrtho;
		pageVueChantier = assistant.pageVueChantier;
		pageVueNuages = assistant.pageVueNuages;
		pageVueHomologues = assistant.pageVueHomologues;
		pageDrawSegment = assistant.pageDrawSegment;
		pageDrawPlanCorrel = assistant.pageDrawPlanCorrel;
		pageInterfOptions = assistant.pageInterfOptions;
		pageInterfVerifMicmac = assistant.pageInterfVerifMicmac;
	}
	return *this;
}

void Assistant::showDocumentation(const QString &page) {
	if (!startAssistant())
		return;
	QByteArray ba("SetSource ");
	ba.append("qthelp://com.trolltech.interfaceMicmac/doc/");

	proc->write(ba + page.toLocal8Bit() + '\0');
}

bool Assistant::startAssistant() {
	if (!proc)
		proc = new QProcess();
	if (proc->state() != QProcess::Running) {
		#if ELISE_windows
			QString app = QLatin1String( ( applicationPath()+QString ("/../interface/help/assistant.exe") ).toStdString().c_str() );
		#else
			QString app = QLatin1String( ( applicationPath()+QString ("/../interface/help/assistant") ).toStdString().c_str() );
		#endif
		QStringList args;
		args << QLatin1String("-collectionFile")
		     << QLatin1String( (applicationPath()+QString("/../interface/help/help.qhc")).toStdString().c_str() )
		     << QLatin1String("-enableRemoteControl");
		proc->start(app, args);
		if (!proc->waitForStarted()) {
			QMessageBox::critical(0, QObject::tr("Interface Micmac"), conv(QObject::tr("Fail to load help file (%1)")).arg(app));
			return false;
		}    
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


QString Options::micmacInstallFile = QString("MicMacConfig.xml");
Options::Options() : micmacDir(QString()), langue(QLocale::French), cpu(Interface::getCpuCount()), cameras(QList<pair<QString,double> >()) {}
Options::Options(const Options& options) { copie(options); }
Options::Options(const QSettings& settings) : cameras(QList<pair<QString,double> >()) {
	setLangue(settings.value("langue").toString());
	setMicmacDir(settings.value("dossierMicmac").toString());
	setCpu(settings.value("cpu").toInt());
}
Options::~Options() {}

Options& Options::operator=(const Options& options) {
	if (&options!=this)
		copie(options);
	return *this;
}

void Options::copie(const Options& options) {
	micmacDir = options.getMicmacDir();
	langue = options.getLangue();
	cpu = options.getCpu();
	cameras = options.getCameras();
}

const QString& Options::getMicmacDir() const { return micmacDir; }
const QLocale::Language& Options::getLangue() const { return langue; }
int Options::getCpu() const { return cpu; }
const QList<std::pair<QString,double> >& Options::getCameras() const { return cameras; }
QList<std::pair<QString,double> >& Options::modifCameras() { return cameras; }

void Options::setMicmacDir(const QString& m) { micmacDir = m; }
void Options::setLangue(const QString& l) {
	if (l==QLocale::languageToString(QLocale::English))
		langue = QLocale::English;
	else
		langue = QLocale::French;
}
void Options::setCpu(int c) { cpu = c; }

bool Options::updateSettings(QSettings& settings) const {
	settings.setValue("dossierMicmac", micmacDir);
	settings.setValue("cpu", cpu);
	settings.setValue("langue", QLocale::languageToString(langue));	//on ne peut pas faire de traduction dynamique, il faudrait renommer chaque label de chaque classe
	BDCamera bdCamera;
	return bdCamera.ecrire(cameras);
}

QStringList Options::checkBinaries(const QString& micmacDossier) {	//micmacDossier doit finir par 1 "/"
	QStringList l;

	if (!QDir(micmacDossier).exists())
		l.push_back( conv(QObject::tr("Selected micmac directory %1 does not exist.")).arg(micmacDossier) );
	else if (!checkPath(micmacDossier))
		l.push_back( conv(QObject::tr("Selected directory %1 contains special characters ; this can provide issues with some calculations. Please modify parent directory name.")).arg(micmacDossier) );

	else {
		if (!QDir(micmacDossier+QString("include")).exists())
			l.push_back( conv(QObject::tr("Directory %1include not found.")).arg(micmacDossier) );
		else if (!QDir(micmacDossier+QString("include/XML_GEN")).exists())
			l.push_back( conv(QObject::tr("Directory %1include/XML_GEN not found.")).arg(micmacDossier) );	
		else if (!QFile(micmacDossier+QString("include/XML_GEN/ParamChantierPhotogram.xml")).exists())
			l.push_back( conv(QObject::tr("File %1include/XML_GEN/ParamChantierPhotogram.xml not found.")).arg(micmacDossier) );

		if (!QDir(micmacDossier+QString("bin")).exists())
			l.push_back( conv(QObject::tr("Directory %1bin not found.")).arg(micmacDossier) );	
		else {
			QString OSextension;
			#if defined Q_WS_WIN 
				OSextension = QString(".exe");
			#endif
			QStringList b("ElDcraw");
                        b << "Devlop" << "tiff_info" << "MapCmd" << "MpDcraw" << "Pastis" << "test_ISA0" << "Apero" << "Bascule" << "Tarama" << "MICMAC" << "GrShade" << "Porto" << "Nuage2Ply";
			QStringList b2;
                        #if defined Q_WS_MAC
                               b2 << "siftpp_tgi.OSX" << "ann_samplekey200filtre.OSX";
                        #elif defined Q_OS_LINUX
                               b2 << "siftpp_tgi.LINUX" << "ann_mec_filtre.LINUX";
                        #elif (defined Q_WS_WIN || defined Q_WS_X11)
                               b2 << "siftpp_tgi" << "ann_samplekeyfiltre";
                        #endif
			for (int i=0; i<b.count(); i++) {
				if (!QFile(micmacDossier+QString("bin/")+b.at(i)+OSextension).exists())
					l.push_back( conv(QObject::tr("Binary %1bin/%2 not found.")).arg(micmacDossier).arg(b.at(i)+OSextension) );
			}
			for (int i=0; i<b2.count(); i++) {
				if (!QFile(micmacDossier+QString("binaire-aux/")+b2.at(i)+OSextension).exists())
					l.push_back( conv(QObject::tr("Binary %1binaire-aux/%2 not found.")).arg(micmacDossier).arg(b2.at(i)+OSextension) );
			}
		}
	}
	return l;
}

bool Options::readMicMacInstall(QString& micmacDossier, int& cpuLu) {
	// try to read Micmac install file if there are info we need
	cpuLu = QThread::idealThreadCount();
	return true;
	
	// we can no longer consider micmac being launched from source directory
	/*
	//lecture du fichier de paramètres ; retourne false s'il y a un problème
	if (!QFile(micmacDossier+micmacInstallFile).exists()) return false;
	cMicMacConfiguration aMF = StdGetObjFromFile<cMicMacConfiguration> ( (micmacDossier+micmacInstallFile).toStdString(), StdGetFileXMLSpec("ParamChantierPhotogram.xml"), "MicMacConfiguration", "MicMacConfiguration" );
	
	if (QString(aMF.DirInstall().c_str())!=micmacDossier) b = false;
	if ( micmacDossier=="" ) micmacDossier=QString(aMF.DirInstall().c_str());
	
	if ( cpuLu<1 ) cpuLu = aMF.NbProcess();

	return b;
	*/
}

void Options::writeMicMacInstall(const QString& micmacDossier, int cpuFinal) {
	/*
	cMicMacConfiguration aMCf;
	aMCf.DirInstall() = micmacDossier.toStdString();
	aMCf.NbProcess() = cpuFinal;
	MakeFileXML(aMCf,(micmacDossier+ELISE_CAR_DIR+micmacInstallFile).toStdString());
	*/
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


InterfOptions::InterfOptions(QWidget* parent, Assistant* help, const QSettings& settings) : QDialog(parent), options(settings), assistant(help), optChanged(false) {
	setWindowModality(Qt::ApplicationModal);
	oldlanguage = options.getLangue();

	//dossier micmac
	QLabel* micmacLabel = new QLabel(conv(tr("Binary directory micmac : ")));
	micmacEdit = new QLineEdit;
	micmacEdit->setMinimumWidth(150);
	micmacEdit->setEnabled(false);
	micmacEdit->setText(options.getMicmacDir());
	micmacButton = new QPushButton(tr("..."));
	micmacButton->setToolTip(conv(tr("Select directory")));
	micmacButton->setMaximumSize (QSize(21,16));

	QHBoxLayout *micmacLayout = new QHBoxLayout;
	micmacLayout->addWidget(micmacLabel);
	micmacLayout->addWidget(micmacEdit);
	micmacLayout->addWidget(micmacButton);
	micmacLayout->addStretch();

	QGroupBox* micmacBox = new QGroupBox;
	micmacBox->setFlat(true);
	micmacBox->setAlignment(Qt::AlignLeft);
	micmacBox->setLayout(micmacLayout);

	//langue
	QLabel* langueLabel = new QLabel(tr("Language : "));
	langueCombo = new QComboBox;
	langueCombo->setMinimumWidth(150);
	langueCombo->addItem(QLocale::languageToString(QLocale::English));
	langueCombo->addItem(QLocale::languageToString(QLocale::French));
	langueCombo->setCurrentIndex(langueCombo->findText(QLocale::languageToString(options.getLangue())));
	langueCombo->setToolTip(conv(tr("Select GUI language")));

	QHBoxLayout *langueLayout = new QHBoxLayout;
	langueLayout->addWidget(langueLabel);
	langueLayout->addWidget(langueCombo);
	langueLayout->addStretch();

	QGroupBox* langueBox = new QGroupBox;
	langueBox->setFlat(true);
	langueBox->setAlignment(Qt::AlignLeft);
	langueBox->setLayout(langueLayout);

	//nombre maximal de processeurs à utiliser
	QLabel* cpuLabel = new QLabel(tr("Max cpu : "));
	cpuSpin = new QSpinBox;
	cpuSpin->setMaximum(Interface::getCpuCount());	//est calculé dès l'ouverture de l'interface et avant l'ouverture de la fenêtre des options
	cpuSpin->setMinimum(1);
	cpuSpin->setValue(options.getCpu());
	cpuSpin->setMaximumWidth(50);
	cpuSpin->setMinimumWidth(50);
	cpuSpin->setToolTip(conv(tr("Maximal number of processors to use for processing")));

	QHBoxLayout *cpuLayout = new QHBoxLayout;
	cpuLayout->addWidget(cpuLabel);
	cpuLayout->addWidget(cpuSpin);
	cpuLayout->addStretch();

	QGroupBox* cpuBox = new QGroupBox;
	cpuBox->setFlat(true);
	cpuBox->setAlignment(Qt::AlignLeft);
	cpuBox->setLayout(cpuLayout);

	//base de données de caméras
	BDCamera bdCamera;
	QString err = bdCamera.lire(options.modifCameras());
	if (!err.isEmpty())
		qMessageBox(this, conv(tr("Read error")), err);
	QLabel* camLabel = new QLabel(conv(tr("Camera data base : ")));
	camList = new QTreeWidget;
	camList->setColumnCount(2);
	camList->setHeaderLabels (QStringList(tr("Name"))<<tr("Pixel (%1m)").arg(QChar(956)));
	camList->setPalette(QPalette(QColor(255,255,255)));
	camList->setMinimumHeight(25);
	if (options.getCameras().count()>0) {
		for (int i=0; i<options.getCameras().count(); i++) {
			QTreeWidgetItem* twi = new QTreeWidgetItem(QStringList(options.getCameras().at(i).first)<<QVariant(options.getCameras().at(i).second).toString());
			camList->addTopLevelItem(twi);
		}
	}
	camList->setSelectionMode (QAbstractItemView::ExtendedSelection);
	camList->adjustSize();

	addCam = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"), QString());
	addCam->setToolTip(conv(tr("Add camera into the data base")));
	addCam->setMaximumSize (QSize(40,34));
	removeCam = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic.png"), QString());
	removeCam->setToolTip(conv(tr("Remove camera from the data base")));
	removeCam->setMaximumSize (QSize(40,34));

	QGridLayout *camLayout = new QGridLayout;
	camLayout->addWidget(camLabel,0,0,1,3);
	camLayout->addWidget(camList,1,0,1,1);
	camLayout->addWidget(addCam,1,1,1,1);
	camLayout->addWidget(removeCam,1,2,1,1);

	QGroupBox* camBox = new QGroupBox;
	camBox->setFlat(true);
	camBox->setAlignment(Qt::AlignLeft);
	camBox->setLayout(camLayout);	

	//boutons
	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	okButton = buttonBox->addButton (tr("Accept"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	assistant = help;
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(micmacBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(langueBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(cpuBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(camBox,0,Qt::AlignHCenter);
	mainLayout->addSpacing(25);
	mainLayout->addWidget(buttonBox);
	mainLayout->addStretch();

	setLayout(mainLayout);
	setWindowTitle(conv(tr("GUI settings")));
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	connect(micmacButton, SIGNAL(clicked()), this, SLOT(micmacClicked()));
	connect(langueCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(langueClicked()));
	connect(cpuSpin, SIGNAL(valueChanged(int)), this, SLOT(cpuClicked()));
	connect(addCam, SIGNAL(clicked()), this, SLOT(addCamClicked()));
	connect(removeCam, SIGNAL(clicked()), this, SLOT(removeCamClicked()));
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(okButton, SIGNAL(clicked()), this, SLOT(okClicked()));
}

InterfOptions::~InterfOptions() {}

void InterfOptions::micmacClicked() {
	//sélection du dossier micmac
	FileDialog fileDialog(this, conv(tr("Micmac directory")), options.getMicmacDir());
	fileDialog.setFilter(QDir::Dirs);
	fileDialog.setFileMode(QFileDialog::Directory);
	fileDialog.setOption(QFileDialog::ShowDirsOnly);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList dirNames;
	if (fileDialog.exec()) {
		dirNames = fileDialog.selectedFiles();
	} else return;
  	if (dirNames.size()!=1)
		return;
  	if (dirNames.at(0)==micmacEdit->text())
		return;
	QString dir = QDir(dirNames.at(0)).absolutePath()+QString("/");
	//vérification des exécutables
	QStringList err = options.checkBinaries(dir);
	if (err.count()>0) {
		for (int i=0; i<err.count(); i++)
			qMessageBox(this, conv(tr("Parameter error")), err.at(i));
	}
	micmacEdit->setText(dir);
	options.setMicmacDir(micmacEdit->text());
	optChanged = true;	//NB : le dossier micmac est validé même si tous les binaires n'y sont pas
}

void InterfOptions::langueClicked() {
	options.setLangue(langueCombo->currentText());
}

void InterfOptions::cpuClicked() {
	options.setCpu(cpuSpin->value());
	optChanged = true;
}

void InterfOptions::addCamClicked() {	
	bool ok;
	QString nom = QInputDialog::getText(this, conv(tr("New camera")), conv(tr("Camera name :")),
					QLineEdit::Normal, QString(), &ok, Qt::Dialog);
	if (!ok) return;
	double px = QInputDialog::getDouble(this, conv(tr("New camera")), conv(tr("Pixel size (%1m) :")).arg(QChar(956)),
					0, 0, 100, 10, &ok, Qt::Dialog);
	if (!ok) return;
	if (px==0) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("Pixel size can not be zero.")));
		return;
	}
	QTreeWidgetItem* twi = new QTreeWidgetItem(QStringList(nom)<<QVariant(px).toString());
	camList->addTopLevelItem(twi);	
}
void InterfOptions::removeCamClicked() {
	QList<QTreeWidgetItem> l;
	for (int i=0; i<camList->topLevelItemCount(); i++) {
		if (!camList->topLevelItem(i)->isSelected())
			l.push_back(*(camList->topLevelItem(i)));
	}
	camList->clear();
	for (int i=0; i<l.count(); i++) {
		QTreeWidgetItem* twi = new QTreeWidgetItem;
		*twi = l.at(i);
		camList->addTopLevelItem(twi);
	}
	camList->adjustSize();
}

void InterfOptions::okClicked() {
	//langue
	if (oldlanguage!=options.getLangue())
		qMessageBox(this, tr("Information"), conv(tr("The language switch will be considered at the next startup of the GUI.")));
	//MicMacConfig.xml
	if (optChanged) options.writeMicMacInstall(options.getMicmacDir(), options.getCpu());	//NB : le dossier micmac est validé même si les paramètres de micmac n'ont pas été enregistrés		
	//caméras
	options.modifCameras().clear();
	for (int i=0; i<camList->topLevelItemCount(); i++) {
		options.modifCameras().push_back(pair<QString,double>(camList->topLevelItem(i)->text(0),camList->topLevelItem(i)->text(1).toDouble()));
	}
	accept();
	hide();
}

void InterfOptions::helpClicked() { assistant->showDocumentation(assistant->pageInterfOptions); }

const Options& InterfOptions::getOptions() const { return options; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


InterfVerifMicmac::InterfVerifMicmac(QWidget* parent, Assistant* help, const ParamMain* param, const QProgressDialog* pDialog) :
	QDialog( parent ),
	assistant( help ),
	paramMain( param ),
	progressDialog( pDialog )
{
	setWindowModality(Qt::ApplicationModal);

	apercuButton = new QToolButton;
	apercuButton2 = new QToolButton;

	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	majButton = new QPushButton;
	majButton->setIcon(QIcon(g_iconDirectory+"update.png"));
	buttonBox->addButton (majButton, QDialogButtonBox::ActionRole);
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(apercuButton,0,Qt::AlignHCenter);
	mainLayout->addWidget(apercuButton2,0,Qt::AlignHCenter);
	mainLayout->addSpacing(25);
	mainLayout->addWidget(buttonBox);
	mainLayout->addStretch();

	setLayout(mainLayout);
	setWindowTitle(conv(tr("Depth map checkout")));
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	connect(majButton, SIGNAL(clicked()), this, SLOT(majClicked()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	majClicked();
}

InterfVerifMicmac::~InterfVerifMicmac() {}

void InterfVerifMicmac::majClicked() {
	//exécution
	int idx = floor((progressDialog->value()-1)/59.);
	int numCarte = -1;
	while (numCarte<paramMain->getParamMicmac().count()-1 && idx>-1) {
		numCarte++;
		if (paramMain->getParamMicmac().at(numCarte).getACalculer()) idx--;
	}
	bool ok;
	QString numRef = paramMain->getNumImage( paramMain->getParamMicmac().at(numCarte).getImageDeReference(), &ok, false );
	if (!ok) {
		cout << tr("Fail to extract reference image number.").toStdString() << endl;
		return;
	}
	QString dir = paramMain->getDossier() + QString("Geo%1%2").arg(paramMain->getParamMicmac().at(numCarte).getRepere()? QString("I") : QString("Ter")).arg(numRef) + QString("/");
	if (QFile(dir+QString("tempo.tif")).exists()) QFile(dir+QString("tempo.tif")).remove();
	QString commande = comm(QString("%1bin/GrShade %2Z_Num1_DeZoom32_Geom-Im-%3.tif Mask=Masq_Geom-Im-%3_DeZoom32.tif Dequant=1 Out=%2tempo.tif >%2tempofile.txt").arg(noBlank(paramMain->getMicmacDir())).arg(noBlank(dir)).arg(numRef));
	if (execute(commande)!=0)  {
		cout << tr("Fail to compute depth map preview.").toStdString() << endl;
		return;
	}
	//conversion en tif non tuilé
	if (QFile(dir+QString("tempo2.tif")).exists()) QFile(dir+QString("tempo2.tif")).remove();
	QString commande2 = comm(QString("%1/lib/tiff2rgba %2tempo.tif %2tempo2.tif").arg(noBlank(applicationPath())).arg(noBlank(dir)));
	if (execute(commande2)!=0) {
		cout << tr("Fail to convert temporary depth map into untiled tif format.").toStdString() << endl;
		QFile(dir+QString("tempo.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		return;
	}
	if (QFile(dir+QString("tempo3.tif")).exists()) QFile(dir+QString("tempo3.tif")).remove();
	QString commande3 = comm(QString("%1/lib/tiff2rgba %2Correl_Geom-Im-%3_Num_1.tif %2tempo3.tif").arg(noBlank(applicationPath())).arg(noBlank(dir)).arg(numRef));
	if (execute(commande3)!=0) {
		cout << tr("Fail to convert image for correlation into untiled tif format.").toStdString() << endl;
		QFile(dir+QString("tempo.tif")).remove();
		QFile(dir+QString("tempo2.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		return;
	}
	//aperçu
	QImage image(dir+QString("tempo2.tif"));
	if (image.isNull()) {
		QFile(dir+QString("tempo.tif")).remove();
		QFile(dir+QString("tempo2.tif")).remove();
		QFile(dir+QString("tempo3.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		cout << tr("Fail to open depth map preview.").toStdString() << endl;
		return;
	}
	image = image.scaled(150,150,Qt::KeepAspectRatio);
	apercuButton->setIconSize(image.size());
	apercuButton->setIcon(QPixmap::fromImage(image));
	apercuButton->adjustSize();

	QImage image2(dir+QString("tempo3.tif"));
	if (image2.isNull()) {
		cout << tr("Fail to open image for correlation.").toStdString() << endl;
		QFile(dir+QString("tempo.tif")).remove();
		QFile(dir+QString("tempo2.tif")).remove();
		QFile(dir+QString("tempo3.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		return;
	}
	image2 = image2.scaled(150,150,Qt::KeepAspectRatio);
	apercuButton2->setIconSize(image2.size());
	apercuButton2->setIcon(QPixmap::fromImage(image2));
	apercuButton2->adjustSize();
	adjustSize();	

	//nettoyage
	QFile(dir+QString("tempo.tif")).remove();
	QFile(dir+QString("tempo2.tif")).remove();
	QFile(dir+QString("tempo3.tif")).remove();
	QFile(dir+QString("tempofile.txt")).remove();
	cout << "ok\n";
	return;
}
void InterfVerifMicmac::helpClicked() { assistant->showDocumentation(assistant->pageInterfVerifMicmac); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Nettoyeur::Nettoyeur() : paramMain(0) {}
Nettoyeur::Nettoyeur(const Nettoyeur& nettoyeur) { copie(this,nettoyeur); }
Nettoyeur::Nettoyeur(const ParamMain* param) : paramMain(param) {}
Nettoyeur::~Nettoyeur() {}

Nettoyeur& Nettoyeur::operator=(const Nettoyeur& nettoyeur) {
	if (this!=&nettoyeur) copie(this,nettoyeur);
	return *this;
}

void copie(Nettoyeur* nettoyeur1, const Nettoyeur& nettoyeur2) {
	nettoyeur1->paramMain = nettoyeur2.paramMain;
}

void Nettoyeur::nettoie() const {
	QStringList l;
	switch (paramMain->getCurrentMode()) {
		case ParamMain::EndMode :
			deleteFile(paramMain->getDossier()+QString("Cartes_A_Creer.xml"));
			deleteFile(paramMain->getDossier()+QString("DefMasque.xml"));
			deleteFile(paramMain->getDossier()+QString("grShade_outstream"));
			deleteFile(paramMain->getDossier()+QString("Image_Maitresse0.xml"));
			deleteFile(paramMain->getDossier()+QString("micmac_outstream.txt"));
			deleteFile(paramMain->getDossier()+QString("micmacTA_outstream.txt"));
			deleteFile(paramMain->getDossier()+QString("Nom_carte.xml"));
			deleteFile(paramMain->getDossier()+QString("Nom_TA.xml"));
			deleteFile(paramMain->getDossier()+QString("Ortho.xml"));
			deleteFile(paramMain->getDossier()+QString("param-GeoImTA.xml"));
			deleteFile(paramMain->getDossier()+QString("param-Terrain.xml"));

			l = QDir(paramMain->getDossier()).entryList(QStringList("TA*"), QDir::Dirs);
			for (int i=0; i<l.count(); i++) {
				deleteFile(paramMain->getDossier()+l.at(i)+QString("/Tmp-MM-Dir/"),true);
				QString numCarte = l.at(i).right(l.at(i).count()-2);
				QStringList l2 = QDir(paramMain->getDossier()+l.at(i)).entryList(QDir::Files);
				for (int j=0; j<l2.count(); j++) {
					if (l2.at(j)==QString("TA_Geom-Im-%1.tif").arg(numCarte)) continue;
					if (l2.at(j)==QString("Z_Num5_DeZoom8_Geom-Im-%1.xml").arg(numCarte)) continue;
					deleteFile(paramMain->getDossier()+l.at(i)+QString("/")+l2.at(j));
				}
			}

			l = QDir(paramMain->getDossier()).entryList(QStringList("Geo*"), QDir::Dirs);
			for (int i=0; i<l.count(); i++) {
				deleteFile(paramMain->getDossier()+l.at(i)+QString("/Tmp-MM-Dir/"),true);
				QString numCarte = (l.at(i).left(4)==QString("GeoI"))? l.at(i).right(l.at(i).count()-4) : l.at(i).right(l.at(i).count()-6);
				QStringList l2 = QDir(paramMain->getDossier()+l.at(i)).entryList(QDir::Files);
				for (int j=0; j<l2.count(); j++) {
					if (l2.at(j).left(15)==QString("Correl_Geom-Im-")) continue;
					if (l2.at(j).left(13)==QString("Masq_Geom-Im-")) continue;
					if (l2.at(j).left(20)==QString("NuageImProf_Geom-Im-")) continue;
					if (l2.at(j).left(5)==QString("Z_Num")) continue;
					deleteFile(paramMain->getDossier()+l.at(i)+QString("/")+l2.at(j));
				}
			}				

		case ParamMain::CarteEnCours :
		case ParamMain::PoseMode :
			deleteFile(paramMain->getDossier()+QString("Apero.xml"));
			deleteFile(paramMain->getDossier()+QString("CleCalibClassiq.xml"));
			deleteFile(paramMain->getDossier()+QString("CleCalibFishEye.xml"));
			deleteFile(paramMain->getDossier()+QString("CleCalibLiberClassiq.xml"));
			deleteFile(paramMain->getDossier()+QString("CleCalibLiberFishEye.xml"));
			deleteFile(paramMain->getDossier()+QString("CleDissocCalib.xml"));
			deleteFile(paramMain->getDossier()+QString("DefCalibrationToutInit.xml"));
			deleteFile(paramMain->getDossier()+QString("DefCalibration.xml"));
			deleteFile(paramMain->getDossier()+QString("DefInconnuesGPS.xml"));
			deleteFile(paramMain->getDossier()+QString("DefObservationsGPS.xml"));
			deleteFile(paramMain->getDossier()+QString("ExportPly.xml"));
			deleteFile(paramMain->getDossier()+QString("Filtrage.xml"));
			deleteFile(paramMain->getDossier()+QString("OrientationAbsolue.xml"));
			deleteFile(paramMain->getDossier()+QString("OrientationGPS.xml"));
			deleteFile(paramMain->getDossier()+QString("OrientationPlanDir.xml"));
			deleteFile(paramMain->getDossier()+QString("PonderationsGPS.xml"));
			deleteFile(paramMain->getDossier()+QString("PosesNonDissoc.xml"));
			deleteFile(paramMain->getDossier()+QString("PosesFigees.xml"));
			deleteFile(paramMain->getDossier()+QString("PosesLibres.xml"));

			l = QDir(paramMain->getDossier()).entryList(QStringList("Ori*"), QDir::Dirs);
			for (int i=0; i<l.count(); i++)
				if (l.at(i)!=QString("Ori-F")) deleteFile(paramMain->getDossier()+l.at(i)+QString("/"), true);

		case ParamMain::PoseEnCours :
		case ParamMain::PointsMode :
			deleteFile(paramMain->getDossier()+QString("Make_CmdExePar"));
			deleteFile(paramMain->getDossier()+QString("MK%1").arg(paramMain->getDossier().section("/",-2,-2)));
			deleteFile(paramMain->getDossier()+QString("MK%1b").arg(paramMain->getDossier().section("/",-2,-2)));
			deleteFile(paramMain->getDossier()+QString("pastis_outstream.txt"));
			deleteFile(paramMain->getDossier()+QString("tempofile.txt"));
			deleteFile(paramMain->getDossier()+QString("timer"));

			l = QDir(paramMain->getDossier()+QString("Pastis")).entryList(QStringList("*.dat")<<"*.tif", QDir::Files);
			for (int i=0; i<l.count(); i++)
				deleteFile(paramMain->getDossier()+QString("Pastis/")+l.at(i));
		default :
			break;
	}
}
