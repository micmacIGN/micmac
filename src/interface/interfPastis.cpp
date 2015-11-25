#if defined Q_WS_WIN 
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "interfPastis.h"

using namespace std;


int InterfPastis::numChoiceTab = 1;

InterfPastis::InterfPastis( QWidget* parent, Assistant* help, ParamMain* pMain):
	QDialog( parent ),
	assistant( help ),
	done( false ),
	paramMain( pMain ),
	paramPastis( paramMain->getParamPastis() ),
	dossier( paramMain->getDossier() ),
	correspImgCalib( &( paramMain->getCorrespImgCalib() ) ),
	longueur( 0 )
{
	setWindowModality(Qt::ApplicationModal);

	//liste des calibrations à fournir
	QApplication::setOverrideCursor(Qt::WaitCursor);
	cout << tr("Used lens search...").toStdString() << endl;

	QList<int>	 calibAFournir;			// a list of focal lengths in mm
	QList<QSize> formatCalibAFournir;	// a list of image sizes in pixel
	QList<int>	 refImgCalibAFournir;	// a list of image indices

	int f;
	QSize s;

	for (int i=0; i<paramMain->getCorrespImgCalib().count(); i++) {
		QString img = paramMain->getCorrespImgCalib().at(i).getImageRAW();
		
		if (!QFile(dossier+img).exists()) img = paramMain->getCorrespImgCalib().at(i).getImageTif();
		if (!QFile(dossier+img).exists()) continue;

		if (img.section(".",-1,-1).toUpper()==QString("TIFF") || img.section(".",-1,-1).toUpper()==QString("TIF")) {
			if ( !focaleTif( dossier+img, paramMain->getMicmacDir(), &f, &s ) )
			{
				cout << tr("Fail to extract image %1 focal length and size.").arg(img).toStdString() << endl;
				continue;
			}
			if ( !calibAFournir.contains(f) )
			{
				calibAFournir.push_back(f);
				formatCalibAFournir.push_back(s);
				refImgCalibAFournir.push_back(i);
			}
		} else if (img.section(".",-1,-1).toUpper()==QString("JPG") || img.section(".",-1,-1).toUpper()==QString("JPEG")) {
			if ( focaleOther( dossier.toStdString(), img.toStdString(), f, s ) )
			{
				if ( !calibAFournir.contains(f) )
				{
					calibAFournir.push_back(f);
					formatCalibAFournir.push_back(s);
					refImgCalibAFournir.push_back(i);
				}
			}
			else
			{
				cout << tr("Fail to extract image %1 focal length and size.").arg(img).toStdString() << endl;
				continue;
			}
		} else {	//raw -> ElDcraw
			QString commande = QString("%1bin/ElDcraw -i -v %2 >%3truc").arg(noBlank(paramMain->getMicmacDir())).arg(noBlank(dossier+img)).arg(noBlank(paramMain->getDossier()));
			if (execute(commande)!=0) {
				cout << tr("Fail to extract image %1 focal length and size.").arg(img).toStdString() << endl;
				continue;
			}
			QFile file(paramMain->getDossier()+QString("truc"));
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
				cout << tr("Fail to extract image %1 focal length and size.").arg(img).toStdString() << endl;
				continue;
			}
			QTextStream inStream(&file);
			int f, larg, lgr;
			f = larg = lgr = 0;
			while (!inStream.atEnd()) {
				QString text = inStream.readLine();
				if (text.contains("Focal length:")) {	//Focal length: 35.0 mm
					QString str = text.simplified().section(" ",2,2);
					bool ok = false;
					f = str.toDouble(&ok);
					if (!ok)  {
						f = 0;
						continue;
					}
				} else if (text.contains("Image size:")) {
					QString str = text.simplified().section(" ",2,2);
					bool ok = false;
					lgr = str.toDouble(&ok);
					if (!ok) {
						lgr = 0;
						continue;
					}
					str = text.simplified().section(" ",4,4);
					larg = str.toDouble(&ok);
					if (!ok) {
						larg = 0;
						continue;
					}
					paramMain->modifCorrespImgCalib()[i].setTaille(QSize(lgr,larg));
				} else continue;
			}
			file.close();
			file.remove();
			/*cMetaDataPhoto metaDataPhoto = cMetaDataPhoto::CreateExiv2((dossier+img).toStdString());
			int f = metaDataPhoto.FocMm();*/
			if (f!=0 && larg!=0 && lgr!=0 && !calibAFournir.contains(f)) {
				calibAFournir.push_back(f);
				formatCalibAFournir.push_back(QSize(lgr,larg));
				refImgCalibAFournir.push_back(i);
			} else if (f==0 || larg==0 || lgr==0)
				cout << tr("Fail to extract image %1 focal length and size.").arg(img).toStdString() << endl;
		}
	}
	longueur = numeric_limits<int>::max();
	if (formatCalibAFournir.count()>0) {
		for (int i=0; i<formatCalibAFournir.count(); i++) {
			if (longueur>formatCalibAFournir.at(i).width()) longueur = formatCalibAFournir.at(i).width();
			else if (longueur>formatCalibAFournir.at(i).height()) longueur = formatCalibAFournir.at(i).height();
		}
	}// else longueur = 0;
	QApplication::restoreOverrideCursor();		

	tabWidget = new QTabWidget;
	tabWidget->setMovable (false);

	calibTab = new CalibTab (this, &paramPastis, paramMain, calibAFournir, formatCalibAFournir, refImgCalibAFournir);//tifOnly, 
	choiceTab = new ChoiceTab (this, &paramPastis);
	coupleTab = new CoupleTab (this, paramMain, &paramPastis);
	if (!coupleTab->isDone()) return;
	communTab = new CommunTabP (this, &paramPastis, longueur);	//par défaut

	tabWidget->addTab(calibTab, conv(tr("Camera")));
	tabWidget->addTab(choiceTab, tr("Survey type"));
	if (paramPastis.getTypeChantier()==ParamPastis::Convergent) tabWidget->addTab(coupleTab, conv(tr("Image pair selection")));
	tabWidget->addTab(communTab, conv(tr("Tie point computation")));

	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	precButton = buttonBox->addButton (conv(tr("Previous")), QDialogButtonBox::ActionRole);
	calButton = buttonBox->addButton (tr("Next"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(tabWidget,0,Qt::AlignCenter);
	mainLayout->addWidget(buttonBox,0,Qt::AlignBottom);
	mainLayout->addStretch();

	setLayout(mainLayout);
	setWindowTitle(conv(tr("Tie point computing parameters")));
	connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged()));
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(precButton, SIGNAL(clicked()), this, SLOT(precClicked()));
	connect(choiceTab, SIGNAL(typechanged()), this, SLOT(typeChantierChanged()));

	oldTypeChantier = ParamPastis::Convergent;
	tabWidget->setCurrentIndex(0);
	//updateInterfPastis(0);
	tabChanged();
	calibTab->resizeTab();
	done = true;
}
InterfPastis::~InterfPastis() {}

void InterfPastis::relaxSize() {
	setMinimumSize(0,0);
	setMaximumSize(16777215,16777215);
}

void InterfPastis::adjustSizeToContent() {
	tabWidget->adjustSize();
	adjustSize();
	setMinimumSize(size());
	setMaximumSize(size());
}

void InterfPastis::tabChanged() {
	relaxSize();
	updateInterfPastis(tabWidget->currentIndex());
	tabWidget->currentWidget()->adjustSize();
	adjustSizeToContent();
}

void InterfPastis::updateInterfPastis(int tab) {
	disconnect(calButton, SIGNAL(clicked()), this, SLOT(suivClicked()));
	disconnect (calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
	calibTab->updateTab(false);

	if (tab==0) {
		precButton->setEnabled(false);
		precButton->setVisible(false);
	} else {
		precButton->setEnabled(true);
		precButton->setVisible(true);
	}

	if (tab==tabWidget->count()-1) {
		calButton->setText ( tr("Compute") );
		connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
	} else {
		calButton->setText ( tr("Next") );
		connect(calButton, SIGNAL(clicked()), this, SLOT(suivClicked()));
	}

	if (tab==tabWidget->indexOf(communTab)) communTab->updateMultiscale();
}
void InterfPastis::precClicked() { tabWidget->setCurrentIndex(tabWidget->currentIndex()-1); }
void InterfPastis::suivClicked() { tabWidget->setCurrentIndex(tabWidget->currentIndex()+1); }

void InterfPastis::typeChantierChanged() {
	if (paramPastis.getTypeChantier()!=oldTypeChantier) {
		if (paramPastis.getTypeChantier()!=ParamPastis::Convergent)
			tabWidget->removeTab(tabWidget->indexOf(coupleTab));
		else
			tabWidget->insertTab(2, coupleTab, tr("Image pair selection"));
		oldTypeChantier = paramPastis.getTypeChantier();	
	}
}

void InterfPastis::calcClicked() {		
	//fichiers de calibration
	if (paramPastis.getCalibFiles().count()==0) {
		qMessageBox(this, conv(tr("Missing parameter")), conv(tr("No calibration file loaded.")));
		return;
	}
	if (!calibTab->allCalibProvided()) {
		qMessageBox(this, conv(tr("Missing parameter")), conv(tr("A calibration file is missing.")));
		return;
	}

	//couples d'images
	if (paramPastis.getTypeChantier()==ParamPastis::Convergent) {
		if (paramPastis.getCouples().count()==0) {	
			qMessageBox(this, conv(tr("No parameters")), conv(tr("No image pairs selected.")));
			return;
		}
		sort(paramPastis.modifCouples().begin(),paramPastis.modifCouples().end());
	} else {
		coupleTab->getAllCouples();
	}

	//largeur maximale
	if (!communTab->getLargeurMaxText(1)) return;
	if (!communTab->getLargeurMaxText(2)) return;

	//fin
	hide();
	accept();
}

void InterfPastis::helpClicked() { 
	if (tabWidget->currentWidget()==calibTab) assistant->showDocumentation(assistant->pageInterPastis);
	else if (tabWidget->currentWidget()==choiceTab) assistant->showDocumentation(assistant->pageInterPastisChoice);
	else if (tabWidget->currentWidget()==coupleTab) assistant->showDocumentation(assistant->pageInterPastisCouple);
	else if (tabWidget->currentWidget()==communTab) assistant->showDocumentation(assistant->pageInterPastisCommun);
}

const ParamPastis& InterfPastis::getParamPastis() const {return paramPastis;}

bool InterfPastis::isDone() { return done; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CalibCam::CalibCam():
	file( QString() ),
	type( 0 ),
	focale( 0 ),
	taillePx( 0 ),
	PPA( QPointF(-1,-1) ),
	sizeImg( QSize(0,0) ),
	PPS( QPointF(-1,-1) ),
	distorsion( QVector<double>(4,0) ),
	rayonUtile( 0 ),
	paramRadial( QVector<double>() ){}

CalibCam::CalibCam(int modele, const QString& fileName, double f, double px, const QPointF& PPApt, const QSize& size, const QPointF& PPSpt, const QVector<double>& dist, int rUtil, const QVector<double>& pRad):
	file( fileName ),
	type( modele ),
	focale( f ),
	taillePx( px ),
	PPA( PPApt ),
	sizeImg( size ),
	PPS( PPSpt ),
	distorsion( dist ),
	rayonUtile( rUtil ),
	paramRadial( pRad ){}
CalibCam::CalibCam(const QString& fileName) : file(fileName) {}
CalibCam::CalibCam(const CalibCam& calibCam) { copie(calibCam); }
CalibCam::~CalibCam() {}

bool CalibCam::operator==(const CalibCam& calibCam) const { return (calibCam.getFile()==file); }
bool CalibCam::operator==(const QString& fileName) const { return (fileName==file); }

CalibCam& CalibCam::operator=(const CalibCam& calibCam) {
	if (&calibCam!=this)
		copie(calibCam);
	return *this;
}

void CalibCam::copie(const CalibCam& calibCam) {
	type = calibCam.getType();
	file = calibCam.getFile();
	focale = calibCam.getFocale();
	taillePx = calibCam.getTaillePx();
	PPA = calibCam.getPPA();
	sizeImg = calibCam.getSizeImg();
	PPS = calibCam.getPPS();
	distorsion = calibCam.getDistorsion();
	rayonUtile = calibCam.getRayonUtile();
	paramRadial = calibCam.getParamRadial();
}

bool CalibCam::setDefaultParam (const QSize& tailleImg) {
	if (sizeImg.width()==0) sizeImg.setWidth(tailleImg.width());
	if (sizeImg.height()==0) sizeImg.setHeight(tailleImg.height());
	if (PPA.x()<=0 || PPA.x()>sizeImg.width()) PPA.setX(sizeImg.width()/2.0);
	if (PPA.y()<=0 || PPA.y()>sizeImg.height()) PPA.setY(sizeImg.height()/2.0);
	if (PPS.x()<=0 || PPS.x()>sizeImg.width()) PPS.setX(sizeImg.width()/2.0);
	if (PPS.y()<=0 || PPS.y()>sizeImg.height()) PPS.setY(sizeImg.height()/2.0);
	if (rayonUtile<=0) rayonUtile = (int)( sqrt( (double)( sizeImg.width()*sizeImg.width()+sizeImg.height()*sizeImg.height() ) )/2. );	//demi-diagonale
	return true;
}

const QString& CalibCam::getFile() const {return file;}
int CalibCam::getType() const {return type;}
double CalibCam::getFocale() const {return focale;}
double CalibCam::getFocalePx() const {return focale * 1000 / taillePx;}
double CalibCam::getTaillePx() const {return taillePx;}
const QPointF& CalibCam::getPPA() const {return PPA;}
const QSize& CalibCam::getSizeImg() const {return sizeImg;}
const QPointF& CalibCam::getPPS() const {return PPS;}
const QVector<double>& CalibCam::getDistorsion() const {return distorsion;}
int CalibCam::getRayonUtile() const {return rayonUtile;}
const QVector<double>& CalibCam::getParamRadial() const {return paramRadial;}

void CalibCam::setFocale (int f) { focale = f; }

QString CalibTab::defaultCalibName = QObject::tr("Calibration_Interne_");
CalibCam CalibTab::defaultCalib = CalibCam();
CalibTab::CalibTab(InterfPastis* interfPastis, ParamPastis* parametres, ParamMain* pMain, const QList<int>& focales, const QList<QSize>& formatCalibs, const QList<int>& refImgCalibs):
	parent( interfPastis ),
	paramPastis( parametres ),
	dir( pMain->getDossier() ),
	paramMain( pMain ),
	calibAFournir( focales ),
	formatCalibAFournir( formatCalibs ),
	refImgCalibAFournir( refImgCalibs ), //bool tifOnly
	imgNames( QList<pair<QString,double> >() ),
	longueur( 0 )
{
	//objectifs utilisés
	QString obj = (calibAFournir.count()==0)? conv(tr("Used lens :")) : conv(tr("Used lenses :"));
	for (int i=0; i<calibAFournir.count(); i++)
		obj += QString ("%1 %2 mm").arg((i==0)? "" : " -").arg(calibAFournir.at(i));
	QLabel* objLabel = new QLabel(obj);
	
	//fichiers de calibration
	calibViewSelected = new QListWidget;
	calibViewSelected->setResizeMode(QListView::Adjust);
	calibViewSelected->setMinimumHeight(25);
	if (paramPastis->getCalibFiles().count()>0) {
		for (int i=0; i<paramPastis->getCalibFiles().count(); i++) {
			if (calibViewSelected->findItems(paramPastis->getCalibFiles().at(i).first,Qt::MatchExactly).size()==0)
				calibViewSelected->addItem(paramPastis->getCalibFiles().at(i).first);
		}
	}
	calibViewSelected->setSelectionMode (QAbstractItemView::ExtendedSelection);
	//calibViewSelected->adjustSize();

	addCalib = new QPushButton(QIcon(g_iconDirectory+"linguist-fileopen.png"), QString());
	addCalib->setToolTip(tr("Load a file"));
	addCalib->setMaximumSize (QSize(40,34));
	removeCalib = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic.png"), QString());
	removeCalib->setToolTip(tr("Remove a file"));
	removeCalib->setMaximumSize (QSize(40,34));
	newCalib = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"), QString());
	newCalib->setToolTip(conv(tr("Create a new file")));
	newCalib->setMaximumSize (QSize(40,34));
	
		//mise en page
		QGridLayout *calibLayout = new QGridLayout;
		calibLayout->addWidget(objLabel,0,0,1,3,Qt::AlignCenter);
		calibLayout->addWidget(calibViewSelected,1,0,1,3,Qt::AlignCenter);
		calibLayout->addWidget(addCalib,2,0,Qt::AlignRight);
		calibLayout->addWidget(removeCalib,2,1);
		calibLayout->addWidget(newCalib,2,2,Qt::AlignLeft);
	
	radioClassique = new QRadioButton(tr("Conical (classic) lens"));
	radioFishEye = new QRadioButton(tr("Fish-eye lens"));
	radioClassique->setChecked(true);
	radioFishEye->setChecked(false);

	QLabel *focaleLabel = new QLabel(tr("Focal length (mm) *"));
	focaleEdit = new QLineEdit;
	//focaleEdit->setInputMask (QString("D00000"));
	//focaleEdit->clear();
	focaleEdit->setToolTip(tr("Required field"));

	//liste des caméras de la BD
	BDCamera bdCamera;
	QString erreur = bdCamera.lire(imgNames);
	if (!erreur.isEmpty()) {
		qMessageBox(this, tr("Read error"),erreur);		
	}

	QLabel *taillePxLabel = new QLabel(tr("Pixel size in ")+QString(QChar(956))+QString("m *"));
	taillePxEdit = new QLineEdit;
	taillePxEdit->setToolTip(tr("Required field"));
	QLabel *tailleCapteurLabel = new QLabel(tr("or Sensor size in mm"));
	tailleCapteurEdit = new QLineEdit;
	QLabel *camLabel = new QLabel(tr("or Camera"));
	camCombo = new QComboBox;
	camCombo->setMinimumWidth(100);
	camCombo->addItem(QString("- - -"));
	if (imgNames.count()>0) {
		for (int i=0; i<imgNames.count(); i++)
			camCombo->addItem(imgNames.at(i).first);
	}

	QLabel *PPALabel = new QLabel(tr("Autocollimation main point"));
	PPAXEdit = new QLineEdit;
	PPAYEdit = new QLineEdit;

	QLabel *sizeLabel = new QLabel(tr("Image size"));
	sizeWEdit = new QLineEdit;
	//sizeWEdit->setInputMask (QString("D00000"));
	//sizeWEdit->clear();
	sizeHEdit = new QLineEdit;
	//sizeHEdit->setInputMask (QString("D00000"));
	//sizeHEdit->clear();

			//paramètres de distorsion pour un objectif classique
		QLabel *PPSLabel = new QLabel(tr("Distorsion center"));
		PPSXEdit = new QLineEdit;
		PPSYEdit = new QLineEdit;

		QLabel *distorsionLabel = new QLabel(tr("Distorsion"));
		distorsionaEdit = new QLineEdit;
		distorsionaEdit->setMinimumWidth(100);
		distorsionbEdit = new QLineEdit;
		distorsionbEdit->setMinimumWidth(100);
		distorsioncEdit = new QLineEdit;
		distorsioncEdit->setMinimumWidth(100);
		distorsiondEdit = new QLineEdit;
		distorsiondEdit->setMinimumWidth(100);

			//paramètres de distorsion pour un objectif fish-eye
		QLabel *rayonLabel = new QLabel(tr("Effective radius"));
		rayonEdit = new QLineEdit;

		QLabel *paramRadialLabel = new QLabel(conv(tr("Distorsion parameters")));
		paramRadialEdit = new QTextEdit;
		paramRadialEdit->setMinimumWidth(100);
		paramRadialEdit->setMinimumHeight(50);

			//mise en boîtes
		QHBoxLayout *PPSLayout = new QHBoxLayout;
		PPSLayout->addWidget(PPSXEdit);
		PPSLayout->addWidget(PPSYEdit);
		PPSLayout->addStretch();

		QHBoxLayout *distorsionLayout = new QHBoxLayout;
		distorsionLayout->addWidget(distorsionaEdit);
		distorsionLayout->addWidget(distorsionbEdit);
		distorsionLayout->addWidget(distorsioncEdit);
		distorsionLayout->addWidget(distorsiondEdit);

		classiqueBox = new QGroupBox;
		classiqueBox->setFlat(true);
		QFormLayout* classiqueLayout = new QFormLayout(classiqueBox);
		classiqueLayout->addRow(PPSLabel,PPSLayout);
		classiqueLayout->addRow(distorsionLabel,distorsionLayout);

		fishEyeBox = new QGroupBox;
		fishEyeBox->setFlat(true);
		QFormLayout* fishEyeLayout = new QFormLayout(fishEyeBox);
		fishEyeLayout->addRow(rayonLabel,rayonEdit);
		fishEyeLayout->addRow(paramRadialLabel,paramRadialEdit);
		
	saveNewCalib = new QPushButton(QIcon(g_iconDirectory+"linguist-filesave.png"), QString());
	saveNewCalib->setToolTip(tr("Save this camera"));
	saveNewCalib->setMaximumSize (QSize(40,34));
	cancelNewCalib = new QPushButton(QIcon(g_iconDirectory+"linguist-prev.png"), QString());
	cancelNewCalib->setMaximumSize (QSize(40,34));
	cancelNewCalib->setToolTip(tr("Cancel"));
	
		//mise en page
		QHBoxLayout *typeBox = new QHBoxLayout;
		typeBox->addWidget(radioClassique);
		typeBox->addWidget(radioFishEye);
		typeBox->addStretch(1);

		QHBoxLayout *pxLayout = new QHBoxLayout;
		pxLayout->addWidget(taillePxEdit);
		pxLayout->addWidget(tailleCapteurLabel);
		pxLayout->addWidget(tailleCapteurEdit);
		if (imgNames.count()>0) pxLayout->addWidget(camLabel);
		if (imgNames.count()>0) pxLayout->addWidget(camCombo);
		pxLayout->addStretch();

		QHBoxLayout *PPALayout = new QHBoxLayout;
		PPALayout->addWidget(PPAXEdit);
		PPALayout->addWidget(PPAYEdit);
		PPALayout->addStretch();

		QHBoxLayout *sizeLayout = new QHBoxLayout;
		sizeLayout->addWidget(sizeWEdit);
		sizeLayout->addWidget(sizeHEdit);
		sizeLayout->addStretch();

		formCalibBox = new QGroupBox;
		formCalibBox->setFlat(true);
		QFormLayout* formLayout = new QFormLayout(formCalibBox);
		formLayout->addRow(typeBox);
		formLayout->addRow(focaleLabel,focaleEdit);
		formLayout->addRow(taillePxLabel,pxLayout);
		formLayout->addRow(PPALabel,PPALayout);
		formLayout->addRow(sizeLabel,sizeLayout);
		formLayout->addRow(classiqueBox);
		formLayout->addRow(fishEyeBox);
		classiqueBox->setVisible(true);
		fishEyeBox->setVisible(false);

		QHBoxLayout* buttonCalibLay = new QHBoxLayout;
		buttonCalibLay->addWidget(saveNewCalib, Qt::AlignRight);
		buttonCalibLay->addWidget(cancelNewCalib, Qt::AlignLeft);
		formLayout->addRow(buttonCalibLay);
		calibLayout->addWidget(formCalibBox,3,0,1,3);

		calibBox = new QGroupBox;
		calibBox->setFlat(false);
		calibBox->setLayout(calibLayout);
		calibBox->setTitle(conv(tr("Camera calibrations : ")));

	QVBoxLayout* mainLayout = new QVBoxLayout();
	mainLayout->addWidget(calibBox, 0, Qt::AlignHCenter);
	mainLayout->addStretch();
	setLayout(mainLayout);	
	updateTab(false,false);

	connect(addCalib, SIGNAL(clicked()), this, SLOT(addCalibClicked()));
	connect(removeCalib, SIGNAL(clicked()), this, SLOT(removeCalibClicked()));
	connect(newCalib, SIGNAL(clicked()), this, SLOT(newCalibClicked()));
	connect(radioClassique, SIGNAL(clicked()), this, SLOT(radioClicked()));
	connect(radioFishEye, SIGNAL(clicked()), this, SLOT(radioClicked()));
	connect(saveNewCalib, SIGNAL(clicked()), this, SLOT(saveNewCalibClicked()));
	connect(cancelNewCalib, SIGNAL(clicked()), this, SLOT(cancelNewCalibClicked()));
	connect(focaleEdit, SIGNAL(textChanged(QString)), this, SLOT(focaleEdited()));
	connect(camCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(camChanged()));
	connect(taillePxEdit, SIGNAL(textChanged(QString)), this, SLOT(taillePxEdited()));
	connect(tailleCapteurEdit, SIGNAL(textChanged(QString)), this, SLOT(tailleCapteurEdited()));
}
CalibTab::~CalibTab() {}

void CalibTab::resizeTab() {
	parent->relaxSize();	
	//calibViewSelected->adjustSize();
	//calibViewSelected->setMinimumWidth(width()-50);

	/*calibViewSelected->setMinimumWidth(0);*/
	adjustSize();
	parent->adjustSizeToContent();
}

void CalibTab::updateTab(bool show, bool resize) {
	if (resize) parent->relaxSize();
	if (show) {
		formCalibBox->show();
	} else {
		formCalibBox->hide();
		focaleEdit->clear();
		PPAXEdit->clear();
		PPAYEdit->clear();
		sizeWEdit->clear();
		sizeHEdit->clear();
		PPSXEdit->clear();
		PPSYEdit->clear();
		distorsionaEdit->clear();
		distorsionbEdit->clear();
		distorsioncEdit->clear();
		distorsiondEdit->clear();
	}
	adjustSize();
	if (resize) parent->adjustSizeToContent();
}

void CalibTab::cancelNewCalibClicked() {updateTab(false);}
void CalibTab::newCalibClicked() {updateTab(true);}
void CalibTab::radioClicked() {
	if (radioClassique->isChecked()) {
		fishEyeBox->setVisible(false);
		classiqueBox->setVisible(true);
	} else  {
		classiqueBox->setVisible(false);
		fishEyeBox->setVisible(true);
	}
}

void CalibTab::removeCalibClicked() {
	updateTab(false);
	if (calibViewSelected->count()==0) return;
	QList<QListWidgetItem *> listItem = calibViewSelected->selectedItems ();
	if (listItem.size()==0) return;
	QStringList l;
	for (int i=0; i<listItem.count(); i++)
		l.push_back(listItem.at(i)->text());	
	removeFromList(l);

	for (int i=0; i<l.count(); i++) {
		for (int j=0; j<paramPastis->getCalibFiles().count(); j++) {
			if (paramPastis->modifCalibFiles().at(j).first==l.at(i)) {
				paramPastis->modifCalibFiles().removeAt(j);
				break;
			}
		}
	}
}

void CalibTab::addCalibClicked() {
	updateTab(false);
	//boîte de dialogue
	FileDialog fileDialog(this, tr("Open a calibration file"), dir, tr("XML files (*.xml)") );
	fileDialog.setFileMode(QFileDialog::ExistingFiles);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec())
		fileNames = fileDialog.selectedFiles();
	else return;
  	if (fileNames.size()==0)
		return;

	//on vérifie que le nom du dossier est lisible (à cause des accents)
	if (!checkPath(fileNames.at(0).section('/', 0, -2))) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read directory.\nCheck there are no accents in path.")));	
		return;
	}

	//lecture et affichage des fichiers
	for (QStringList::const_iterator it=fileNames.begin(); it!=fileNames.end(); it++) {
		//extraction du dossier et du nom du fichier
		QString fichier = *it;

		//on vérifie que le nom du fichier est lisible (à cause des accents)
		if (!checkPath(fichier)) {
			qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
			return;
		}

		QString dossier = fichier.section('/', 0, -2);
		dossier = QDir(dossier).absolutePath() + QString("/");
		QString xmlFile = fichier.section('/', -1, -1);

		//focale
		int focale;
		if (!paramPastis->extractFocaleFromName(xmlFile, focale)) {
			qMessageBox(this, tr("Read error"), conv(tr("Fail to extract focal length from calibration file name %1.\nPlease rename this file (focale length must be indicated at the end of the name).")).arg(fichier));
			continue;
		}

		//on vérifie que c'est bien un fichier de calibration lisible
		CalibCam calibCam;
		QString err = FichierCalibCam::lire(dossier, xmlFile, calibCam, focale);
		if (!err.isEmpty()) {
			qMessageBox(this, tr("Read error"), err);
			continue;
		}

		//on vérifie qu'il n'y a pas déjà une calibration de cette focale dans la liste
		QString oldfile;
		int pos = -1;
		for (int i=0; i<paramPastis->getCalibFiles().count(); i++) {
			if (paramPastis->getCalibFiles().at(i).second==focale) {
				oldfile = paramPastis->getCalibFiles().at(i).first;
				pos = i;
				break;
			}
		}
		if (!oldfile.isEmpty()) {
			int rep = Interface::dispMsgBox( conv(tr("A calibration file already exists for focal length %1.").arg(focale)),
							conv(tr("Do you want to update the calibration parameters ?")),
							QVector<int>()<<0<<1<<-1, 0);
			if (rep==0) {
				QFile(dir+oldfile).remove();
				paramPastis->modifCalibFiles().removeAt(pos);
				removeFromList(QStringList(oldfile));
			} else
				continue;
		}

		//renommination et copie
		QString newFile = dir+xmlFile.section(".",0,-2)+QString(".xml");
		if (fichier!=newFile && QFile(newFile).exists()) {
			int rep = Interface::dispMsgBox( conv(tr("File %1 already exists.").arg(newFile)),
							conv(tr("Do you want to replace it by file %1 ?").arg(fichier)),
							QVector<int>()<<0<<1<<-1, 0);
			if (rep==0) QFile(newFile).remove();
			else continue;
		}
		if (dossier!=dir) QFile(fichier).copy(newFile);	//on conserve le fichier de l'utilisateur
		else if (fichier!=newFile) QFile(fichier).rename(newFile);	//c'était pas la bonne extension

		//enregistrement et affichage
		paramPastis->modifCalibFiles().push_back( pair<QString, int>(xmlFile.section(".",0,-2)+QString(".xml"), focale) );
		calibViewSelected->addItem(xmlFile.section(".",0,-2)+QString(".xml"));
		calibViewSelected->adjustSize();
	}
}

double CalibTab::qstringToDouble(QString coeff, bool* ok) {	//traduit distorsionEdit en double
	double q=0;
	if (ok!=0) *ok = true;
	QString s(coeff);
	s.replace(QString(","),QString("."));
	s.replace(QString("*10^"),QString("e"));
	s.replace(QString(".10^"),QString("e"));
	s.remove(QString("+"));
	s.remove(QString("*"));
	s.simplified();
	s.remove(QString(" "));

	QString smanti = s;
	double expos = 0.0;
	if (s.contains(QString("e"))) {
		smanti = s.section(QString("e"),0,0);		
		expos = s.section(QString("e"),-1,-1).toDouble(ok);	
		if (ok!=0 && !*ok) return 0;
	}

	double manti = smanti.toDouble(ok);	
		if (ok!=0 && !*ok) return 0;
	q = manti * pow(10, expos);
	return q;
}
void CalibTab::focaleEdited() {
	if (focaleEdit->text().isEmpty()) return;
	longueur = 0;
	bool ok = false;
	int f = focaleEdit->text().toInt(&ok);
	if (!ok) return;
	for (int i=0; i<calibAFournir.count(); i++) {
		if (calibAFournir.at(i)==f) {
			longueur = max(formatCalibAFournir.at(i).width(), formatCalibAFournir.at(i).height());
			break;
		}
	}
	if (longueur==0) return;
	taillePxEdited();	//pour relancer le calcul
}
void CalibTab::taillePxEdited() {
	disconnect(tailleCapteurEdit, SIGNAL(textChanged(QString)), this, SLOT(tailleCapteurEdited()));
	disconnect(camCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(camChanged()));
	if (taillePxEdit->text().isEmpty() || longueur==0) {
		tailleCapteurEdit->clear();
	} else {
		bool ok = false;
		double px = taillePxEdit->text().toDouble(&ok);
		if (!ok) return;
		tailleCapteurEdit->setText(QVariant(int(px*longueur/1000)).toString());
	}
	camCombo->setCurrentIndex(0);	//pas de caméra
	connect(tailleCapteurEdit, SIGNAL(textChanged(QString)), this, SLOT(tailleCapteurEdited()));
	connect(camCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(camChanged()));
}
void CalibTab::tailleCapteurEdited() {
	disconnect(taillePxEdit, SIGNAL(textChanged(QString)), this, SLOT(taillePxEdited()));
	disconnect(camCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(camChanged()));
	if (tailleCapteurEdit->text().isEmpty() || longueur==0) {
		taillePxEdit->clear();
	} else {
		bool ok = false;
		double cam = tailleCapteurEdit->text().toDouble(&ok);
		if (!ok) return;
		if (longueur!=0) taillePxEdit->setText(QVariant(cam/double(longueur)*1000.0).toString());
	}
	camCombo->setCurrentIndex(0);	//pas de caméra
	connect(taillePxEdit, SIGNAL(textChanged(QString)), this, SLOT(taillePxEdited()));
	connect(camCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(camChanged()));
}

void CalibTab::camChanged() {
	disconnect(tailleCapteurEdit, SIGNAL(textChanged(QString)), this, SLOT(tailleCapteurEdited()));
	disconnect(taillePxEdit, SIGNAL(textChanged(QString)), this, SLOT(taillePxEdited()));
	if (camCombo->currentIndex()>0) {
cout << camCombo->currentIndex() << endl;
		double px = imgNames.at(camCombo->currentIndex()-1).second;
cout << px << endl;
		taillePxEdit->setText(QVariant(px).toString());
		if (longueur!=0) tailleCapteurEdit->setText(QVariant(int(px*longueur/1000)).toString());
		else tailleCapteurEdit->clear();
	}
	connect(tailleCapteurEdit, SIGNAL(textChanged(QString)), this, SLOT(tailleCapteurEdited()));
	connect(taillePxEdit, SIGNAL(textChanged(QString)), this, SLOT(taillePxEdited()));
}

void CalibTab::saveNewCalibClicked() {
	bool ok = false;
	if (focaleEdit->text().isEmpty()) {
		qMessageBox(this, conv(tr("No focal length entered")), tr("A focal length must be written."));
		return;
	}
	focaleEdit->text().toInt(&ok);
	if (!ok) {
		qMessageBox(this, conv(tr("Uncorrect value")), tr("The focal length is not an integer."));
		return;
	}
	if (taillePxEdit->text().isEmpty()) {
		qMessageBox(this, conv(tr("No pixel size entered")), tr("A pixel size must be written."));
		return;
	}
	if (!taillePxEdit->text().isEmpty()) {
		taillePxEdit->text().toDouble(&ok);
		if (!ok) {
			qMessageBox(this, conv(tr("Uncorrect value")), tr("The pixel size is not a number."));
			return;
		}
	}
	if (!PPAXEdit->text().isEmpty()) {
		PPAXEdit->text().toDouble(&ok);
		if (!ok) {
			qMessageBox(this, conv(tr("Uncorrect value")), tr("The main point (x coordinate) is not a number."));
			return;
		}
	}
	if (!PPAYEdit->text().isEmpty()) {
		PPAYEdit->text().toDouble(&ok);
		if (!ok) {
			qMessageBox(this, conv(tr("Uncorrect value")), tr("The main point (y coordinate) is not a number."));
			return;
		}
	}
	if (!sizeWEdit->text().isEmpty()) {
		sizeWEdit->text().toInt(&ok);
		if (!ok) {
			qMessageBox(this, conv(tr("Uncorrect value")), tr("The image height is not a number."));
			return;
		}
	}
	if (!sizeHEdit->text().isEmpty()) {
		sizeHEdit->text().toInt(&ok);
		if (!ok) {
			qMessageBox(this, conv(tr("Uncorrect value")), tr("The image width is not a number."));
			return;
		}
	}
	if (radioClassique->isChecked()) {
		if (!PPSXEdit->text().isEmpty()) {
			PPSXEdit->text().toDouble(&ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), tr("The distorsion center (x coordinate) is not a number."));
				return;
			}
		}
		if (!PPSYEdit->text().isEmpty()) {
			PPSYEdit->text().toDouble(&ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), tr("The distorsion center (y coordinate) is not a number."));
				return;
			}
		}
		if (!distorsionaEdit->text().isEmpty()) {
			qstringToDouble(distorsionaEdit->text(), &ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), conv(tr("The first distorsion parameter is not a number.")));
				return;
			}
		}
		if (!distorsionbEdit->text().isEmpty()) {
			qstringToDouble(distorsionbEdit->text(), &ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), conv(tr("The second distorsion parameter is not a number.")));
				return;
			}
		}
		if (!distorsioncEdit->text().isEmpty()) {
			qstringToDouble(distorsioncEdit->text(), &ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), conv(tr("The third distorsion parameter is not a number.")));
				return;
			}
		}
		if (!distorsiondEdit->text().isEmpty()) {
			qstringToDouble(distorsiondEdit->text(), &ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), conv(tr("The last distorsion parameter is not a number.")));
				return;
			}
		}
	} else {
		if (!rayonEdit->text().isEmpty()) {
			rayonEdit->text().toDouble(&ok);
			if (!ok) {
				qMessageBox(this, conv(tr("Uncorrect value")), tr("The efficient radius is not a number."));
				return;
			}
		}
		QString text = paramRadialEdit->toPlainText();
		if (!text.isEmpty()) {
			QString text2 = text.section("\n",0,0);
			int i = 0;
			while(!text2.isEmpty()) {
				qstringToDouble(text2, &ok);
				if (!ok) {
					qMessageBox(this, conv(tr("Uncorrect value")), text2+conv(tr(" :\nThis parameter is not a number.")));
					return;
				}
				i++;
				text2 = text.section("\n",i,i);
			}
		}

	}

	int type = (radioClassique->isChecked())? 0 : 1;
	int focale = focaleEdit->text().toInt();
	int PPAX = (PPAXEdit->text().isEmpty())? defaultCalib.getPPA().x() : PPAXEdit->text().toDouble();
	int PPAY = (PPAYEdit->text().isEmpty())? defaultCalib.getPPA().y() : PPAYEdit->text().toDouble();
	int sizeX = (sizeWEdit->text().isEmpty())? defaultCalib.getSizeImg().width() : sizeWEdit->text().toDouble();
	int sizeY = (sizeHEdit->text().isEmpty())? defaultCalib.getSizeImg().height() : sizeHEdit->text().toDouble();
	int PPSX = (PPSXEdit->text().isEmpty())? defaultCalib.getPPS().x() : PPSXEdit->text().toDouble();
	int PPSY = (PPSYEdit->text().isEmpty())? defaultCalib.getPPS().y() : PPSYEdit->text().toDouble();
	QVector<double> dist(4);	
		dist[0] = (distorsionaEdit->text().isEmpty())? defaultCalib.getDistorsion()[0] : qstringToDouble(distorsionaEdit->text());
		dist[1] = (distorsionbEdit->text().isEmpty())? defaultCalib.getDistorsion()[1] : qstringToDouble(distorsionbEdit->text());
		dist[2] = (distorsioncEdit->text().isEmpty())? defaultCalib.getDistorsion()[2] : qstringToDouble(distorsioncEdit->text());
		dist[3] = (distorsiondEdit->text().isEmpty())? defaultCalib.getDistorsion()[3] : qstringToDouble(distorsiondEdit->text());
	int rayonUtile = (rayonEdit->text().isEmpty())? defaultCalib.getRayonUtile() : rayonEdit->text().toDouble();
	QString text = paramRadialEdit->toPlainText();
	QVector<double> paramRadial(text.count("\n")+1);
		QString text2 = text.section("\n",0,0);
		int i = 0;
		while(!text2.isEmpty()) {
			if (i<paramRadial.count()) paramRadial[i] = text2.toDouble();
			else paramRadial.push_back(text2.toDouble());
			i++;
			text2 = text.section("\n",i,i);
		}	
		paramRadial.resize(i);

	//nom du fichier
	QString sfocale = QVariant(focale).toString();
	while (sfocale.count()<3)
		sfocale.prepend(QString("0"));
	QString file = defaultCalibName + sfocale + QString(".xml");

	//on vérifie qu'il n'y a pas déjà une calibration de cette focale dans la liste
	QString oldfile;
	int pos = -1;
	for (int i=0; i<paramPastis->getCalibFiles().count(); i++) {
		if (paramPastis->getCalibFiles().at(i).second==focale) {
			oldfile = paramPastis->getCalibFiles().at(i).first;
			pos = i;
			break;
		}
	}
	if (!oldfile.isEmpty()) {
		int rep = Interface::dispMsgBox( conv(tr("A calibration file already exists for focal length %1.").arg(focale)),
						conv(tr("Do you want to update the calibration parameters ?")),
						QVector<int>()<<0<<1<<-1, 0);
		if (rep==0) {
			QFile(dir+oldfile).remove();
			paramPastis->modifCalibFiles().removeAt(pos);
			removeFromList(QStringList(oldfile));
		} else
			return;
	}

	//enregistrement
	if (QFile(dir+file).exists()) {
		int rep = Interface::dispMsgBox( conv(tr("File %1 already exists.").arg(file)),
						conv(tr("Do you want to replace it ?")),
						QVector<int>()<<0<<1<<-1, 0);
		if (rep==0) QFile(dir+file).remove();
		else return;
	}
	CalibCam calibCam(type, file, focale, taillePxEdit->text().toDouble(), QPointF(PPAX,PPAY), QSize(sizeX,sizeY), QPointF(PPSX,PPSY), dist, rayonUtile, paramRadial);
	if (!FichierCalibCam::ecrire(dir, calibCam)) {
		qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to save calibration.")));
		return;
	}
	paramPastis->modifCalibFiles().push_back( pair<QString,int>(file,focale) );

	//ajout dans la BD MicMac
	//-- int idxCalib = 0;
	while (i<calibAFournir.count() && calibAFournir.at(i)!=calibCam.getFocale()) i++;
	if (i!=calibAFournir.count() && QFile(paramMain->getDossier()+paramMain->getCorrespImgCalib().at(i).getImageRAW()).exists()) {
		QString extension = paramMain->getCorrespImgCalib().at(i).getImageRAW().section(".",-1,-1);
		if (!extension.toUpper().contains("TIF") && extension.toUpper()!=QString("JPG") && extension.toUpper()!=QString("JPEG")) {
			DicoCameraMM dicoCameraMM(paramMain);
			QString err = dicoCameraMM.ecrire(calibCam,paramMain->getDossier()+paramMain->getCorrespImgCalib().at(i).getImageRAW());
			if (!err.isEmpty()) {
				qMessageBox(this, conv(tr("Write error")), err);
				return;
			}
		}
	}

	//affichage
	calibViewSelected->addItem(file);
	calibViewSelected->adjustSize ();
	updateTab(false);
}

bool CalibTab::allCalibProvided() {
	for (int i=0; i<calibAFournir.count(); i++) {
		bool b = false;
		for (int j=0; j<paramPastis->getCalibFiles().count(); j++) {
			if (paramPastis->getCalibFiles().at(j).second==calibAFournir.at(i)) {
				b = true;
				break;
			}
		}
		if (!b) return false;
	}
	return true;
}

void CalibTab::removeFromList(const QStringList& items) {
//supprime des fichiers items de la listWidget
	if (items.size()==0) return;
	QStringList liste;	
	for (int i=0; i<calibViewSelected->count(); i++)
		liste.push_back( calibViewSelected->item(i)->text() );
	calibViewSelected->clear();

	for (QStringList::const_iterator item=items.begin(); item!=items.end(); item++)
		liste.removeOne(*item);	
	if (liste.size()==0) return;

	for (QStringList::const_iterator item=liste.begin(); item!=liste.end(); item++)
		calibViewSelected->addItem(*item);
	calibViewSelected->adjustSize();
}

const QList<int>& CalibTab::getCalibAFournir() const { return calibAFournir; }
const QList<QSize>& CalibTab::getFormatCalibAFournir() const { return formatCalibAFournir; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ChoiceTab::ChoiceTab(InterfPastis* interfPastis, ParamPastis* parametres) : parent(interfPastis)
{
	paramPastis = parametres;
	radioConvergent = new QRadioButton(tr("&Converging shoot"));
	radioBandes = new QRadioButton(tr("Parallel shoot"));
	//radioAutre = new QRadioButton(tr("Autre type"));
	switch (paramPastis->getTypeChantier()) {
		case ParamPastis::Convergent:
			radioConvergent->setChecked (true);
			break;
		case ParamPastis::Bandes:
			radioBandes->setChecked (true);
			break;
		//case ParamPastis::Autre:
		//	radioAutre->setChecked (true);
		//	break;
	}
	connect(radioConvergent, SIGNAL(clicked()), this, SLOT(radioClicked()));
	connect(radioBandes, SIGNAL(clicked()), this, SLOT(radioClicked()));
	//connect(radioAutre, SIGNAL(clicked()), this, SLOT(radioClicked()));

	QVBoxLayout *vbox = new QVBoxLayout;
	vbox->addWidget(radioConvergent);
	vbox->addWidget(radioBandes);
	//vbox->addWidget(radioAutre);
	vbox->addStretch(1);

	QGroupBox* groupBox = new QGroupBox(this);
	groupBox->setObjectName(conv(tr("Select the shoot type :")));
	groupBox->setLayout(vbox);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(groupBox, 0, Qt::AlignCenter);
	setLayout(mainLayout);
	mainLayout->setAlignment(Qt::AlignCenter);
}
ChoiceTab::~ChoiceTab() {}

void ChoiceTab::radioClicked() {
	if (radioConvergent->isChecked())
		paramPastis->setTypeChantier(ParamPastis::Convergent);
	else if (radioBandes->isChecked())
		paramPastis->setTypeChantier(ParamPastis::Bandes);
	//else if (radioAutre->isChecked())
	//	paramPastis->setTypeChantier(ParamPastis::Autre);
	emit typechanged();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


CoupleTab::CoupleTab(InterfPastis* interfPastis, ParamMain* paramMain, ParamPastis* parametres)
	: parent(interfPastis), paramPastis(parametres), dir(paramMain->getDossier()), correspImgCalib(&(paramMain->getCorrespImgCalib()))
{
	done = false;
	QToolButton* label1 = new QToolButton;
	label1->setText(conv(tr("First image")));
	QToolButton* label2 = new QToolButton;
	label2->setText(tr("Second image"));
	
	listWidget1 = new QListWidget;
	listWidget1->setSelectionMode (QAbstractItemView::ExtendedSelection);
	listWidget2 = new QListWidget;
	listWidget2->setSelectionMode (QAbstractItemView::ExtendedSelection);

	treeWidget = new QTreeWidget;
	treeWidget->setHeaderLabel(conv(tr("Image pairs")));
	treeWidget->setColumnCount(1);
	treeWidget->setSelectionMode (QAbstractItemView::ExtendedSelection);

	paramPastis->modifCouples().clear();
	if (QFile(dir+paramMain->getCoupleXML()).exists()) {
		QString lecture = FichierCouples::lire (dir+paramMain->getCoupleXML(), paramPastis->modifCouples(), paramMain->getCorrespImgCalib());
		if (!lecture.isEmpty()) {
			qMessageBox(this, tr("Read error."), lecture);
			return;
		}
	}
	if (paramPastis->getCouples().count()>0) {
		for (int i=0; i<paramPastis->getCouples().count(); i++) {
			QList<QTreeWidgetItem *> l = treeWidget->findItems(paramPastis->getCouples().at(i).first, Qt::MatchExactly);
			QTreeWidgetItem * treeWidgetItem = (l.count()>0)? *(l.begin()) : new QTreeWidgetItem(QStringList(paramPastis->getCouples().at(i).first));
			if (l.count()==0) treeWidget->addTopLevelItem(treeWidgetItem);
			QTreeWidgetItem* treeChildWidgetItem = new QTreeWidgetItem(QStringList(paramPastis->getCouples().at(i).second));
			treeWidgetItem->addChild(treeChildWidgetItem);	
			treeWidgetItem->sortChildren(0, Qt::AscendingOrder);
		}
		treeWidget->sortItems(0, Qt::AscendingOrder);
	}
	initList(listWidget1);

	addButton = new QToolButton;
	addButton->setIcon(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"));
	addButton->setMaximumSize (QSize(40,34));
	addButton->setToolTip(tr("Add this pair"));
	addAllButton = new QPushButton(QIcon(g_iconDirectory+"linguist-next.png"), QString());
	addAllButton->setMaximumSize (QSize(40,34));
	addAllButton->setToolTip(conv(tr("Select all pairs...")));
	removeButton = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic2.png"), QString());
	removeButton->setMaximumSize (QSize(40,34));
	removeButton->setToolTip(tr("Remove this pair"));
	removeAllButton = new QPushButton(QIcon(g_iconDirectory+"linguist-prev.png"), QString());
	removeAllButton->setMaximumSize (QSize(40,34));
	removeAllButton->setToolTip(tr("Remove all pairs"));
	expandCollapseButton = new QPushButton(QIcon(g_iconDirectory+"collapse.png"), QString());
	expandCollapseButton->setMaximumSize (QSize(40,34));
	expandCollapseButton->setToolTip(conv(tr("Collapse the tree")));
	expandCollapseButton->setEnabled (false);

	addAllAct = new QAction(conv(tr("Select all pairs")), this);
	connect(addAllAct, SIGNAL(triggered()), this, SLOT(addAllClicked()));
	addKNearestAct = new QAction(conv(tr("Select pairs with the k nearest image method")), this);
	connect(addKNearestAct, SIGNAL(triggered()), this, SLOT(addKNearestClicked()));

	QGridLayout *gridLayout = new QGridLayout;
	gridLayout->addWidget(label1,0,0,1,1,Qt::AlignCenter);
	gridLayout->addWidget(listWidget1,1,0,4,1);
	gridLayout->addWidget(label2,0,1,1,1,Qt::AlignCenter);
	gridLayout->addWidget(listWidget2,1,1,4,1);
	gridLayout->addWidget(addButton,0,2,1,1);
	gridLayout->addWidget(addAllButton,1,2,1,1);
	gridLayout->addWidget(removeButton,2,2,1,1);
	gridLayout->addWidget(removeAllButton,3,2,1,1);
	gridLayout->addWidget(expandCollapseButton,4,2,1,1);
	gridLayout->addWidget(treeWidget,0,3,5,1);

	groupBox = new QGroupBox(this);
	groupBox->setTitle(conv(tr("Select image pairs to process :")));
	groupBox->setLayout(gridLayout);
	groupBox->showMaximized();

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(groupBox, 0, Qt::AlignCenter);
	setLayout(mainLayout);

	connect(listWidget1, SIGNAL(itemSelectionChanged()), this, SLOT(list1Selected()));
	connect(listWidget2, SIGNAL(itemSelectionChanged()), this, SLOT(list2Selected()));
	connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(treeWidgetSelected()));
	connect(addButton, SIGNAL(clicked()), this, SLOT(addTheseClicked()));
	connect(addAllButton, SIGNAL(pressed()), this, SLOT(addAllButtonClicked()));
	connect(removeButton, SIGNAL(clicked()), this, SLOT(removeClicked()));	
	connect(removeAllButton, SIGNAL(clicked()), this, SLOT(removeAllClicked()));
	connect(expandCollapseButton, SIGNAL(clicked()), this, SLOT(collapseAll()));

	updateCoupleTab();
	done = true;
}
CoupleTab::~CoupleTab() {
	delete addAllAct;
	delete addKNearestAct;
}

void CoupleTab::initList(QListWidget* listWidget) {
	//ajoute toutes les images (dont les couples ne sont pas tous dans l'arbre) dans la liste 1
	listWidget->clear();
	for (int i=0; i<correspImgCalib->size(); i++) {
		//recherche du nombre de couples correspondants dans l'arbre
		int N = 0;
		for (int j=0; j<treeWidget->topLevelItemCount(); j++) {
			if (treeWidget->topLevelItem(j)->text(0)==correspImgCalib->at(i).getImageRAW())
				N += treeWidget->topLevelItem(j)->childCount();
			else {
				for (int k=0; k<treeWidget->topLevelItem(j)->childCount(); k++) {
					if (treeWidget->topLevelItem(j)->child(k)->text(0)==correspImgCalib->at(i).getImageRAW()) N++;
				}
			}
		}
		if (N<correspImgCalib->size()-1) listWidget->addItem( correspImgCalib->at(i).getImageRAW() );
	}
	listWidget->sortItems();
	listWidget->adjustSize();
}

void CoupleTab::list1Selected() {
	//ajoute toutes les images dans la liste 2 (sauf les couples correspondant à l'image sélectionnée qui sont déjà placés dans l'arbre)
	if (listWidget1->selectedItems().size()>0) {
		addButton->setEnabled(true);
		if (listWidget1->selectedItems().size()==1)
			addButton->setToolTip(tr("Add all pairs related to this image"));
		else
			addButton->setToolTip(tr("Add all pairs related to these images"));
	}
	if (listWidget1->selectedItems().size()!=1)
		return;

	listWidget2->clear();
	//récupération de l'img1
	QString img1 = (*(listWidget1->selectedItems().begin()))->text();
	QList<QTreeWidgetItem *> l = treeWidget->findItems(img1, Qt::MatchExactly);
	QTreeWidgetItem* treeWidgetItem = (l.size()==0)? 0 : *(l.begin());
	
	if (treeWidgetItem!=0 && treeWidgetItem->childCount()==correspImgCalib->count())	//toutes les images ont déjà été ajoutées
		return;

	//ajout des img2 dans la liste 2
	for (int i=0; i<correspImgCalib->size(); i++) {
		if (correspImgCalib->at(i).getImageRAW()<=img1)	//on ne met les couples qu'une fois
			continue;
		int j=0;
		if (treeWidgetItem!=0) {
			//recherche de l'img 2 dans l'arbre
			while (j<treeWidgetItem->childCount() && treeWidgetItem->child(j)->text(0)!=correspImgCalib->at(i).getImageRAW())
				j++;
		}
		//il n'y est pas ou img1 n'y est pas
		if (treeWidgetItem==0 || j==treeWidgetItem->childCount())
			listWidget2->addItem( correspImgCalib->at(i).getImageRAW() );
	}
	listWidget2->sortItems();
	updateCoupleTab();
}

void CoupleTab::list2Selected() {
	if (listWidget2->selectedItems().size()==0)
		return;
	if (listWidget2->selectedItems().size()==1)
		addButton->setToolTip(tr("Add this pair"));
	else
		addButton->setToolTip(tr("Add these pairs"));
}

void CoupleTab::treeWidgetSelected() {
	listWidget2->clear();
	removeButton->setEnabled(true);
}

void CoupleTab::addTheseClicked() {
	//cas 1 : on ajoute tous les couples d'une ou plusieurs images
	if (listWidget2->selectedItems().size()==0) {
		QStringList l1;
		for (int i=0; i<listWidget1->count(); i++) {	//listWidget1->removeItemWidget(item) ne marche pas
			if (!listWidget1->item(i)->isSelected())
				l1.push_back(listWidget1->item(i)->text());
		}

		for (int k=0; k<listWidget1->selectedItems().count(); k++) {
			//récupération de img1
			QString img1 = listWidget1->selectedItems().at(k)->text();
			
			//ajout de img1 comme parent
			bool b = ( listWidget1->row(listWidget1->selectedItems().at(k)) == listWidget1->count()-1 );	//si img1 est la dernière des images dans l'odre alphabétique, elle n'aura pas d'enfant dans l'arbre et il ne faut pas l'ajouter comme parent ; elle est aussi dernière de la liste1. Si elle est dernière de la liste1 mais pas dernière des images, elle existe déjà dans l'arbre comme parent de la dernière des images
			QList<QTreeWidgetItem *> l = treeWidget->findItems(img1, Qt::MatchExactly);
			QTreeWidgetItem* treeWidgetItem = (l.size()==0 && b)? 0 : (l.size()==0)? new QTreeWidgetItem(QStringList(img1)) : *(l.begin());
			if (l.size()==0 && treeWidgetItem!=0)
				treeWidget->addTopLevelItem(treeWidgetItem);

			//ajout de tous les couples associés
			for (int i=0; i<correspImgCalib->count(); i++) {
				QString img2 = correspImgCalib->at(i).getImageRAW();
				if (img2==img1) continue;
				else if (img2>img1) {	//treeWidgetItem!=0
					//ajout de l'img2 comme enfant de img1
					bool exists = false;
					for (int j=0; j<treeWidgetItem->childCount(); j++) {
						if (treeWidgetItem->child(j)->text(0)==img2) {
							exists = true;
							break;
						}
					}
					if (exists) continue;
					QTreeWidgetItem* treeWidgetItemChild2 = new QTreeWidgetItem(QStringList(img2));
					treeWidgetItem->addChild(treeWidgetItemChild2);
					paramPastis->modifCouples().push_back(pair<QString,QString>(img1,img2));
					treeWidgetItem->sortChildren(0, Qt::AscendingOrder);
				} else	{
					//ajout de img1 comme enfant de img2
						//récupération de img2
					QList<QTreeWidgetItem *> l2 = treeWidget->findItems(img2, Qt::MatchExactly);
					QTreeWidgetItem* treeWidgetItem2 = (l2.size()==0)? new QTreeWidgetItem(QStringList(img2)) : *(l2.begin());
					if (l2.size()==0) {
						treeWidget->addTopLevelItem(treeWidgetItem2);
					}
						//ajout de img1
					bool exists = false;
					for (int j=0; j<treeWidgetItem2->childCount(); j++) {
						if (treeWidgetItem2->child(j)->text(0)==img1) {
							exists = true;
							break;
						}
					}
					if (exists) continue;
					QTreeWidgetItem* treeWidgetItemChild = new QTreeWidgetItem(QStringList(img1));
					treeWidgetItem2->addChild(treeWidgetItemChild);
					paramPastis->modifCouples().push_back(pair<QString,QString>(img2,img1));
					treeWidgetItem2->sortChildren(0, Qt::AscendingOrder);
					
				}
			}
			if (treeWidgetItem!=0) treeWidgetItem->sortChildren(0, Qt::AscendingOrder);

			//vérification dans la liste 1
			int n = treeWidget->indexOfTopLevelItem(treeWidgetItem);
			for (int i=0; i<n; i++) {
				if (treeWidget->topLevelItem(i)->childCount()==correspImgCalib->count()-1-i)
					l1.removeOne(treeWidget->topLevelItem(i)->text(0));
			}
		}
		treeWidget->sortItems(0, Qt::AscendingOrder);

		//mise à jour de la liste 1
		listWidget1->clear();
		for (int i=0; i<l1.size(); i++)
			listWidget1->addItem(l1.at(i));
		l1.clear();
	}

	//cas 2 : on ajoute les couples sélectionnés
	else {
		QString img1 = (*(listWidget1->selectedItems().begin()))->text();
		//ajout/récupération de img1
		QList<QTreeWidgetItem *> l = treeWidget->findItems(img1, Qt::MatchExactly);
		QTreeWidgetItem* treeWidgetItem = (l.size()==0)? new QTreeWidgetItem(QStringList(img1)) : *(l.begin());
		if (l.size()==0) {
			treeWidget->addTopLevelItem(treeWidgetItem);
		}

		QList<QString> l2;
		for (int i=0; i<listWidget2->count(); i++) {
			QString img2 = listWidget2->item(i)->text();
			if (listWidget2->item(i)->isSelected()) {
				//ajout des img2
				QTreeWidgetItem* treeWidgetItemChild = new QTreeWidgetItem(QStringList(img2));
				treeWidgetItem->addChild(treeWidgetItemChild);
				paramPastis->modifCouples().push_back(pair<QString,QString>(img1,img2));		
			} else {
				//récupération des img2 non sélectionnées
				l2.push_back(img2);
			}
		}
		treeWidgetItem->sortChildren(0, Qt::AscendingOrder);

		//mise à jour de la liste 2
		listWidget2->clear();
		if (l2.count()>0) {
			for (QList<QString>::const_iterator it=l2.begin(); it!=l2.end(); it++) {
				listWidget2->addItem(*it);
			}
			l2.clear();
		} else {
			//il n'y a plus d'img correspondante, mise à jour de la liste 1
			QList<QString> l1;
			for (int i=0; i<listWidget1->count(); i++) {	//listWidget1->removeItemWidget(item) ne marche pas
				if (!listWidget1->item(i)->isSelected())
					l1.push_back(listWidget1->item(i)->text());
			}
			listWidget1->clear();
			for (int i=0; i<l1.size(); i++)
				listWidget1->addItem(l1.at(i));
			l1.clear();
		}
		treeWidget->sortItems(0, Qt::AscendingOrder);
	}
	
	updateCoupleTab();	
}

void CoupleTab::addAllButtonClicked() {
	//affiche le menu
	QMenu menu(addButton);
	menu.addAction(addAllAct);
	menu.addAction(addKNearestAct);
	menu.exec(addButton->mapToGlobal(QPoint(addButton->width(), 0)));
}

void CoupleTab::addAllClicked() {
	//ajoute tous les couples dans l'arbre
	treeWidget->clear();
	for (int i=0; i<correspImgCalib->size(); i++) {
		QTreeWidgetItem* treeWidgetItem = new QTreeWidgetItem( QStringList(correspImgCalib->at(i).getImageRAW()) );
		for (int j=0; j<correspImgCalib->size(); j++) {
			if (correspImgCalib->at(i).getImageRAW()>=correspImgCalib->at(j).getImageRAW())
				continue;
			QTreeWidgetItem* treeWidgetItemChild = new QTreeWidgetItem( QStringList(correspImgCalib->at(j).getImageRAW()) );
			treeWidgetItem->addChild( treeWidgetItemChild );
			paramPastis->modifCouples().push_back(pair<QString,QString>(correspImgCalib->at(i).getImageRAW(),correspImgCalib->at(j).getImageRAW()));
		}
		if (treeWidgetItem->childCount()==0) {
			delete treeWidgetItem;
			continue;
		}
		treeWidget->addTopLevelItem(treeWidgetItem);
		treeWidgetItem->sortChildren(0, Qt::AscendingOrder);		
	}
	treeWidget->sortItems(0, Qt::AscendingOrder);
	//mise à jour des listes
	listWidget1->clear();
	listWidget2->clear();
	updateCoupleTab();
	treeWidget->expandAll();
}

void CoupleTab::addKNearestClicked() {
	//on ouvre une boîte de dialogue pour sélectionner K (écart maximal entre les numéros des images pour former les couples d'images voisines)
	bool ok;
	int K = QInputDialog::getInt(this, tr("Choose the image neighbourhood parameters"),
					conv(tr("Maximal distance between image index to make a pair :")),
					3, 1, correspImgCalib->size()/2, 1,
					&ok, Qt::Dialog);
	if (!ok) return;

	//on ajoute les couples à l'arbre s'il n'y sont pas déjà
		//sauvegarde des img de la liste 1
	QStringList l1;
	for (int i=0; i<listWidget1->count(); i++) {	//listWidget1->removeItemWidget(item) ne marche pas
		if (!listWidget1->item(i)->isSelected())
			l1.push_back(listWidget1->item(i)->text());
	}

	for (int i=0; i<correspImgCalib->size(); i++) {	//img1
		if (i==correspImgCalib->size()-1) continue; //il n'y a pas d'img2

		//ajout/récupération de l'img1 dans l'arbre
		QList<QTreeWidgetItem *> l = treeWidget->findItems(correspImgCalib->at(i).getImageRAW(), Qt::MatchExactly);
		QTreeWidgetItem* treeWidgetItem = (l.size()==0)? new QTreeWidgetItem(QStringList(correspImgCalib->at(i).getImageRAW())) : *(l.begin());
		if (l.size()==0) {
			treeWidget->addTopLevelItem(treeWidgetItem);
		}
		
		for (int j=i+1; j<min(i+K+1,correspImgCalib->size()); j++) {	//img2 (les K suivantes)
			//ajout/récupération de l'img2 dans l'arbre
			bool exists = false;
			for (int n=0; n<treeWidgetItem->childCount(); n++) {
				if (treeWidgetItem->child(n)->text(0)==correspImgCalib->at(j).getImageRAW()) {
					exists = true;
					break;
				}
			}
			if (!exists) {
				QTreeWidgetItem* treeWidgetItem2 = new QTreeWidgetItem(QStringList(correspImgCalib->at(j).getImageRAW()));
				treeWidgetItem->addChild(treeWidgetItem2);
				paramPastis->modifCouples().push_back(pair<QString,QString>(correspImgCalib->at(i).getImageRAW(),correspImgCalib->at(j).getImageRAW()));
			}
		}

		if (i<K) {	//i est l'img2 des K dernières images (cas des objets dont on fait le tour complet)
			for (int j=correspImgCalib->size()-K+i; j<correspImgCalib->size(); j++) {
				//ajout/récupération de l'img2 dans l'arbre
				bool exists = false;
				for (int n=0; n<treeWidgetItem->childCount(); n++) {
					if (treeWidgetItem->child(n)->text(0)==correspImgCalib->at(j).getImageRAW()) {
						exists = true;
						break;
					}
				}
				if (!exists) {
					QTreeWidgetItem* treeWidgetItem2 = new QTreeWidgetItem(QStringList(correspImgCalib->at(j).getImageRAW()));
					treeWidgetItem->addChild(treeWidgetItem2);
					paramPastis->modifCouples().push_back(pair<QString,QString>(correspImgCalib->at(i).getImageRAW(),correspImgCalib->at(j).getImageRAW()));
				}
			}
		}

		treeWidgetItem->sortChildren(0, Qt::AscendingOrder);
		if (treeWidgetItem->childCount()==correspImgCalib->size()-i-1) {	//tous les couples correspondant à l'img1 ont été sélectionnés
			l1.removeOne(correspImgCalib->at(i).getImageRAW());
		}
	}
	treeWidget->sortItems(0, Qt::AscendingOrder);
	listWidget2->clear();
	//mise à jour de la liste 1
	listWidget1->clear();
	for (int i=0; i<l1.size(); i++)
		listWidget1->addItem(l1.at(i));
	l1.clear();
	
	updateCoupleTab();
}

void CoupleTab::removeClicked() {
	if (treeWidget->selectedItems().size()==0)
		return;

	QStringList l1;	//élément à rajouter à la liste 1 (s'ils n'y étaient pas)
	QList<pair<QString, QStringList> > lt;
	/*for (int i=0; i<treeWidget->topLevelItemCount(); i++) {
		QStringList l;
		for (int j=0; j<treeWidget->topLevelItem(i)->childCount(); j++) {
			l.push_back(treeWidget->topLevelItem(i)->child(j)->text(0));
		}
		lt.push_back(pair<QString, QStringList>(treeWidget->topLevelItem(i)->text(0),l));
	}*/

	for (int i=0; i<treeWidget->topLevelItemCount(); i++) {
		QString img1 = treeWidget->topLevelItem(i)->text(0);
		//cas où les items enfants sont sélectionnés
		for (int j=0; j<treeWidget->topLevelItem(i)->childCount(); j++) {
			if (treeWidget->topLevelItem(i)->child(j)->isSelected()) {
				if (l1.indexOf(img1)==-1) l1.push_back(img1);
				//on les supprime
				paramPastis->modifCouples().removeOne(pair<QString, QString>(img1,treeWidget->topLevelItem(i)->child(j)->text(0)));
				treeWidget->topLevelItem(i)->removeChild(treeWidget->topLevelItem(i)->child(j));
				j--;					
			}
		}
		//cas où l'item parent est sélectionné : on le supprime pour chaque couple (y compris s'il n'y a pas d'enfant correspondant)
		if (treeWidget->topLevelItem(i)->isSelected()) {
			if (paramPastis->getCouples().count()>0) {
				for (int i=0; i<paramPastis->getCouples().count(); i++) {
					if (paramPastis->getCouples().at(i).first==img1 || paramPastis->getCouples().at(i).second==img1) {
						paramPastis->modifCouples().removeAt(i);
						i--;
					}
				}
			}
			if (l1.indexOf(img1)==-1) l1.push_back(img1);
			//recherche dans chaque item parent de l'item enfant img1 et suppression
			for (int j=0; j<treeWidget->topLevelItemCount(); j++) {
				if (treeWidget->topLevelItem(j)->text(0)>=img1) continue;
				QString img2 = treeWidget->topLevelItem(j)->text(0);
				for (int k=0; k<treeWidget->topLevelItem(j)->childCount(); k++) {
					if (treeWidget->topLevelItem(j)->child(k)->text(0)==img1) {
						treeWidget->topLevelItem(j)->removeChild(treeWidget->topLevelItem(j)->child(k));
						paramPastis->modifCouples().removeOne(pair<QString, QString>(img1,treeWidget->topLevelItem(j)->child(k)->text(0)));
						if (l1.indexOf(img2)==-1) l1.push_back(img2);
						break;
					}
				}
			}
		}	
	}
	for (int i=0; i<treeWidget->topLevelItemCount(); i++) {
		//on sauvegarde tous les items qui restent avec leurs items enfants
		if ((!treeWidget->topLevelItem(i)->isSelected()) && treeWidget->topLevelItem(i)->childCount()>0) {
			QList<QString> l2;
			for (int j=0; j<treeWidget->topLevelItem(i)->childCount(); j++) {
				l2.push_back(treeWidget->topLevelItem(i)->child(j)->text(0));
			}			
			lt.push_back(pair<QString, QList<QString> > (treeWidget->topLevelItem(i)->text(0),l2));
		}
	}

	//mise à jour de l'arbre
	treeWidget->clear();
	for (QList<pair<QString, QStringList > >::const_iterator it=lt.begin(); it!=lt.end(); it++) {
		QTreeWidgetItem* treeWidgetItem = new QTreeWidgetItem(QStringList(it->first));
		for (QList<QString>::const_iterator it2=it->second.begin(); it2!=it->second.end(); it2++) {
			QTreeWidgetItem* treeWidgetItemChild = new QTreeWidgetItem(QStringList(*it2));
			treeWidgetItem->addChild(treeWidgetItemChild);
		}
		treeWidget->addTopLevelItem(treeWidgetItem);	//l'ordre est conservé
	}
	lt.clear();

	//mise à jour de la liste 1
	for (QStringList::const_iterator it=l1.begin(); it!=l1.end(); it++) {
		if (listWidget1->findItems(*it, Qt::MatchExactly).count()==0)
			listWidget1->addItem(*it);
	}
	listWidget1->sortItems();
	l1.clear();	

	updateCoupleTab();
}

void CoupleTab::removeAllClicked() {
	treeWidget->clear();
	paramPastis->modifCouples().clear();
	initList(listWidget1);
	listWidget2->clear();
	updateCoupleTab();
}

void CoupleTab::updateCoupleTab () {
	removeButton->setEnabled(false);
	if (listWidget1->selectedItems().count()==0)
		addButton->setEnabled(false);
	if (treeWidget->topLevelItemCount()==0) {
		removeButton->setEnabled(false);
		removeAllButton->setEnabled(false);
		expandCollapseButton->setEnabled(false);
	} else {
		removeAllButton->setEnabled(true);
		expandCollapseButton->setEnabled(true);
	}
	if (listWidget1->count()==0)
		addAllButton->setEnabled(false);
	else
		addAllButton->setEnabled(true);	
	treeWidget->adjustSize ();
	listWidget1->adjustSize();
	listWidget2->adjustSize();
	//groupBox->adjustSize();
	//parent->adjustSizeToContent();
}

void CoupleTab::expandAll() {
	disconnect(expandCollapseButton, SIGNAL(clicked()), this, SLOT(expandAll()));
	connect(expandCollapseButton, SIGNAL(clicked()), this, SLOT(collapseAll()));
	treeWidget->expandAll ();
	expandCollapseButton->setIcon(QIcon(g_iconDirectory+"collapse.png"));
	expandCollapseButton->setToolTip(conv(tr("Collapse the tree")));	
}

void CoupleTab::collapseAll() {
	disconnect(expandCollapseButton, SIGNAL(clicked()), this, SLOT(collapseAll()));
	connect(expandCollapseButton, SIGNAL(clicked()), this, SLOT(expandAll()));
	treeWidget->collapseAll ();	
	expandCollapseButton->setIcon(QIcon(g_iconDirectory+"expand.png"));
	expandCollapseButton->setToolTip(conv(tr("Expand the tree")));	
}

void CoupleTab::getAllCouples() const {
//retourne tous les couples possibles (cas parallèle), indépendamment de ce qui a été saisi
	paramPastis->modifCouples().clear();
	for (int i=0; i<correspImgCalib->count()-1; i++) {
	for (int j=i+1; j<correspImgCalib->count(); j++) {
		paramPastis->modifCouples().push_back(pair<QString, QString>(correspImgCalib->at(i).getImageRAW(),correspImgCalib->at(j).getImageRAW()));
	}}
}
bool CoupleTab::isDone() { return done; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


CommunTabP::CommunTabP(InterfPastis* interfPastis, ParamPastis* parametres, int largeurMax):
	tailleMax( largeurMax ),
	parent( interfPastis ),
	paramPastis( parametres )
{
	//recherche multiéchelle
	checkMultiscale = new QCheckBox(conv(tr("Multiscale computing")));
	checkMultiscale->setChecked(paramPastis->getMultiScale());
	checkMultiscale->setToolTip (conv(tr("If checked, tie-point search is computed in two steps :\nThe first step is a search in all previously selected images,\nthe second step is done in only image pairs where enough tie-points have been found the first time.")));

	//tailles des images rééchantillonnées
	largeurMax1Label = new QLabel();	//recherche mono-échelle ou multi-échelle passe 1
	largeurMax2Label = new QLabel(conv(tr("Maximum width of rescaled image (second step) :")));	//recherche multi-échelle passe 2
	largeurMax1Edit = new QLineEdit;	//recherche mono-échelle ou multi-échelle passe 1
	largeurMax1Edit->setMaximumWidth (100);
	int l1 = paramPastis->getLargeurMax();
	if (paramPastis->getMultiScale() && paramPastis->getTypeChantier()==ParamPastis::Convergent && paramPastis->getCouples().count()<21) l1 = paramPastis->getLargeurMax2();	//cas particulier où c'était un calcul multi-échelle que l'on reprend avec seulement les images conservées
	if (l1==0 || l1>tailleMax) l1 = 1000;
	largeurMax1Edit->setText(QVariant(l1).toString());
	largeurMax1Edit->setToolTip (conv(tr("Tie-point search is done in images that are rescaled at a low resolution.\nWrite -1 for a full size search.")));
	largeurMax2Edit = new QLineEdit;	//recherche multi-échelle passe 2
	int l2 = paramPastis->getLargeurMax2();
	if (l2==0 || l2>tailleMax || !paramPastis->getMultiScale()) l2 = 1000;
	largeurMax2Edit->setMaximumWidth (100);
	largeurMax2Edit->setText(QVariant(l2).toString());
	largeurMax2Edit->setToolTip (conv(tr("Tie-point search is done in images that are rescaled at a low resolution.\nWrite -1 for a full size search.")));

	//seuil pour la 2nde passe
	seuilLabel = new QLabel(conv(tr("Minimum number of tie-points :")));
	seuilBox = new QSpinBox;
	seuilBox->setValue(paramPastis->getNbPtMin());
	seuilBox->setMaximum(1000);
	seuilBox->setMinimum(1);
	seuilBox->setMaximumWidth(50);
	seuilBox->setMinimumWidth(50);
	seuilBox->setToolTip(conv(tr("First step tie-point number threshold to keep an image pair for second step computation.")));

	//mise en page
	QFormLayout *formLayout = new QFormLayout;
	formLayout->addRow(checkMultiscale);
	formLayout->addRow(largeurMax1Label,largeurMax1Edit);
	formLayout->addRow(seuilLabel,seuilBox);
	formLayout->addRow(largeurMax2Label,largeurMax2Edit);
	formLayout->setFormAlignment(Qt::AlignCenter);
	setLayout(formLayout);

	connect(seuilBox, SIGNAL(valueChanged(int)), this, SLOT(seuilChanged()));
	connect(checkMultiscale, SIGNAL(stateChanged(int)), this, SLOT(updateMultiscale()));
	updateMultiscale();
}
CommunTabP::~CommunTabP() {}

void CommunTabP::updateMultiscale() {
	parent->relaxSize();
	if (paramPastis->getTypeChantier()==ParamPastis::Convergent && paramPastis->getCouples().count()<21) {	//chantier convergent avec peu de couples (équivalent all, 5 images), pas de choix => mono-échelle
		checkMultiscale->hide();
		checkMultiscale->setChecked(false);
	} else
		checkMultiscale->show();

	paramPastis->setMultiScale(checkMultiscale->isChecked());
	if (!checkMultiscale->isChecked()) {	//mono-échelle
		largeurMax1Label->setText(conv(tr("Maximal width of rescaled image :")));
		largeurMax2Label->hide();
		largeurMax2Edit->hide();
		seuilLabel->hide();	
		seuilBox->hide();	

	} else {	//multi-échelle
		largeurMax1Label->setText(conv(tr("Maximum width of rescaled image (first step) :")));
		largeurMax2Label->show();
		largeurMax2Edit->show();
		seuilLabel->show();	
		seuilBox->show();		
	}
	adjustSize();
	parent->adjustSizeToContent();
}

void CommunTabP::seuilChanged() { paramPastis->setNbPtMin(seuilBox->value()); }

bool CommunTabP::getLargeurMaxText(int passe) const {
cout << paramPastis->getMultiScale() << endl;
	if (passe==2 && !paramPastis->getMultiScale()) return true;
	QString text = (passe==1) ? largeurMax1Edit->text() : largeurMax2Edit->text();
	if (text.isEmpty()) {
		qMessageBox(parent, conv(tr("No parameters")), conv(tr("A maximum width is required to rescale the images.")));
		return false;
	}
	bool ok = true;
	int lrgrMax = text.toInt(&ok);
	if (!ok || lrgrMax==0) {
		qMessageBox(parent, tr("Uncorrect size"), conv(tr("Uncorrect image width for rescaling.")));	
		return false;	
	}
	if (lrgrMax>tailleMax) {
		qMessageBox(parent, tr("Uncorrect size"), conv(tr("Rescaled image size is higher than full size image size.")));	
		return false;	
	}

	if (passe==1 && paramPastis->getMultiScale() && (lrgrMax==-1 || lrgrMax==tailleMax) ) {
		qMessageBox(parent, tr("Uncorrect size"), conv(tr("Rescaled image size is the same as image size.\nMultiscale search is useless.\nPlease uncheck multiscale search.")));	
		return false;	
	} else if (passe==2 && lrgrMax<=paramPastis->getLargeurMax() && lrgrMax!=-1) {
		qMessageBox(parent, tr("Uncorrect size"), conv(tr("Rescaled image size for second step is shorter than the first step image size.\nMultiscale search is useless.\nPlease uncheck multiscale search.")));	
		return false;	
	}

	if (passe==1) paramPastis->setLargeurMax(lrgrMax);
	else paramPastis->setLargeurMax2(lrgrMax);
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

QVector<std::pair<ParamPastis::TypeChantier,QString> > ParamPastis::tradTypChan(0);
QVector<std::pair<ParamPastis::TypeChantier,QString> > ParamPastis::tradTypChanInternational(0);

ParamPastis::ParamPastis():
	typeChantier( Convergent ),
	calibs( QList<CalibCam>() ),
	calibFiles( QList<pair<QString,int> >() ),
	largeurMax( 0 ),
	couples( QList<pair<QString,QString> >() ),
	multiscale( false ),
	largeurMax2( 0 ),
	nbPtMin( 2 ){ fillTradTypChan(); }
ParamPastis::ParamPastis(const ParamPastis& paramPastis) { fillTradTypChan(); copie(paramPastis); }
ParamPastis::~ParamPastis() {}

ParamPastis& ParamPastis::operator=(const ParamPastis& paramPastis) {
	if (&paramPastis!=this)
		copie(paramPastis);
	return *this;
}

void ParamPastis::copie(const ParamPastis& paramPastis) {
	typeChantier = paramPastis.getTypeChantier();
	calibs = paramPastis.getCalibs();
	calibFiles = paramPastis.getCalibFiles();
	largeurMax = paramPastis.getLargeurMax();
	couples = paramPastis.getCouples();	
	multiscale = paramPastis.getMultiScale();	
	largeurMax2 = paramPastis.getLargeurMax2();	
	nbPtMin = paramPastis.getNbPtMin();	
}

void ParamPastis::fillTradTypChan() {
	if (tradTypChan.count()==0) {
		tradTypChan.resize(2);
		tradTypChan[0] = pair<TypeChantier,QString>(Convergent, conv("Converging shoot")) ;
		tradTypChan[1] = pair<TypeChantier,QString>(Bandes, conv("Parallel shoot")) ;
		//tradTypChan[2] = pair<TypeChantier,QString>(Autre, conv("Autre type de chantier")) ;
	}
	if (tradTypChanInternational.count()==0) {
		tradTypChanInternational.resize(2);
		tradTypChanInternational[0] = pair<TypeChantier,QString>(Convergent, conv(QObject::tr("Converging shoot"))) ;
		tradTypChanInternational[1] = pair<TypeChantier,QString>(Bandes, conv(QObject::tr("Parallel shoot"))) ;
		//tradTypChanInternational[2] = pair<TypeChantier,QString>(Autre, conv(QObject::tr("Autre type de chantier"))) ;
	}
}

int ParamPastis::findFocale(const QString& calibFile) const {
	QList<pair<QString, int> >::const_iterator it=calibFiles.begin();
	while (it!=calibFiles.end()) {
		if (it->first==calibFile) return it->second;
		it++;
	}
	return -1;
}

bool ParamPastis::extractFocaleFromName(const QString& calibFile, int& focale) {
//extrait la focale du nom du fichier de calibration (doit être écrite à la fin du nom hors extension)
	QString name = calibFile.section(".",0,-2);
	int pos = name.count();	//nb de charactères à partir de la fin
	for (int i=0; i<name.count(); i++) {
		if (name.at(name.count()-i-1).category()!=QChar::Number_DecimalDigit) {
			pos = i;
			break;
		}
	}
	bool ok;
	focale = name.right(pos).toInt(&ok);
	return ok;
}

const QVector<pair<ParamPastis::TypeChantier,QString> >& ParamPastis::getTradTypChan() {return tradTypChan;}
const QVector<pair<ParamPastis::TypeChantier,QString> >& ParamPastis::getTradTypChanInternational() {return tradTypChanInternational;}
int ParamPastis::getLargeurMax() const {return largeurMax;}
const QList<CalibCam>& ParamPastis::getCalibs() const {return calibs;}
const QList<std::pair<QString, int> >& ParamPastis::getCalibFiles() const {return calibFiles;}
const ParamPastis::TypeChantier& ParamPastis::getTypeChantier() const {return typeChantier;}
const QList<pair<QString, QString> >& ParamPastis::getCouples() const {return couples;}
bool ParamPastis::getMultiScale() const {return multiscale;}
int ParamPastis::getLargeurMax2() const {return largeurMax2;}
int ParamPastis::getNbPtMin() const {return nbPtMin;}

void ParamPastis::setLargeurMax(int l) {largeurMax=l;}
void ParamPastis::setTypeChantier(const TypeChantier& typChan) {typeChantier = typChan;}
QList<CalibCam>& ParamPastis::modifCalibs() {return calibs;}
QList<std::pair<QString, int> >& ParamPastis::modifCalibFiles() {return calibFiles;}
QList<pair<QString, QString> >& ParamPastis::modifCouples() {return couples;}
void ParamPastis::setMultiScale(bool b) {multiscale=b;}
void ParamPastis::setLargeurMax2(int l) {largeurMax2=l;}
void ParamPastis::setNbPtMin(int n) {nbPtMin=n;}


