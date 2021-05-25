#include "interfConversion.h"

using namespace std;


InterfCartes8B::InterfCartes8B(const QString& dossier, const QString& dossierMicmac, const QList<ParamConvert8B::carte8B>* cartes, QWidget* parent, Assistant* help) : QDialog(parent), dir(dossier), micmacDir(dossierMicmac),  paramConvert8B(ParamConvert8B())
{
	done = false;
	setWindowModality(Qt::ApplicationModal);
	//resize(QApplication::desktop()->availableGeometry().size());

	//images à convertir
	listeCartes = cartes;
	listImages = new QListWidget;
	//listImages->setResizeMode(QListView::Adjust);
	for (QList<ParamConvert8B::carte8B>::const_iterator it=listeCartes->begin(); it!=listeCartes->end(); it++) {
		listImages->insertItem(0,it->getCarte16B());
	}
	listImages->setSelectionMode(QAbstractItemView::ExtendedSelection);
	listImages->item(0)->setSelected(true);	//dézoom=1
	//listImages->setContentsMargins(5,5,5,5);
	QFontMetricsF metrics = QFontMetrics(listImages->font());
	QString text;
	for (int i=0; i<listImages->count(); i++) {
		text += listImages->item(0)->text() + QString("\n");
	}
	QRect maxr(0,0,QApplication::desktop()->availableGeometry().width()/2,QApplication::desktop()->availableGeometry().height()/2);
	QRectF r = metrics.boundingRect(maxr, Qt::AlignLeft|Qt::AlignVCenter, text);
	listImages->setFixedSize(r.width()+40,min(r.height()+40+listImages->count()*metrics.leading(), 200.0));	//ascenseurs de listImages

	//options
	checkVisu = new QCheckBox(tr("Show conversion"));
	checkVisu->setToolTip(conv(tr("Display conversion steps")));
	checkVisu->setChecked(paramConvert8B.visualiser);
	checkMask = new QCheckBox(conv(tr("Hide background")));
	checkMask->setToolTip(conv(tr("Use the mask with suitable zoom to hide scene background")));
	checkMask->setChecked(paramConvert8B.useMasque);
	checkDequant = new QCheckBox(conv(tr("Unquantify")));
	checkDequant->setToolTip(tr("Smooth depth values of depth map"));
	checkDequant->setChecked(paramConvert8B.dequantifier);
	optionButton = new QPushButton(tr("Other options"));

	QVBoxLayout* vBoxLayout = new QVBoxLayout;
	vBoxLayout->addWidget(checkVisu);
	vBoxLayout->addWidget(checkMask);
	vBoxLayout->addWidget(checkDequant);
	vBoxLayout->addWidget(optionButton, 0, Qt::AlignRight);
	QGroupBox* groupBox = new QGroupBox;
	groupBox->setFlat(true);
	groupBox->setLayout(vBoxLayout);

	//options facultatives
		//image
	imageCheck = new QCheckBox(conv(tr("Use a texture image")));
	imageCheck->setChecked(paramConvert8B.withImg);
	imageCheck->setToolTip(conv(tr("If checked, the colour image is used to texture the shaded depth map.")));

		//nb directions
	QLabel* nbdirLabel = new QLabel(tr("Number of light sources"));
	nbdirBox = new QSpinBox;
	nbdirBox->setMaximum(300);
	nbdirBox->setMinimum(1);
	nbdirBox->setValue(paramConvert8B.nbDir);
	nbdirBox->setMaximumWidth(50);
	nbdirBox->setMinimumWidth(50);
	nbdirBox->setToolTip(conv(tr("Number of ligth sources spread throughout the scene.")));

		//mode d'ombrage
	QLabel* modeombreLabel = new QLabel(tr("Shade mode"));
	modeOmbreBox = new QComboBox;
	modeOmbreBox->addItems(QStringList("CielVu")<<"Local"<<"Mixte"<<"Med"<<"IGN espace");	//traduire ?
	modeOmbreBox->setCurrentIndex(paramConvert8B.modeOmbre);
	//nbdirBox->setToolTip(conv(tr("Nombre de sources de lumières réparties autour de la scène.")));

		//relief
	QLabel* fzLabel = new QLabel(tr("Relief"));
	fzEdit = new QLineEdit;
	fzEdit->setText(QVariant(paramConvert8B.fz).toString());
	fzEdit->setMaximumWidth(50);
	fzEdit->setMinimumWidth(50);
	fzEdit->setToolTip(conv(tr("The relief is multiplied by this coefficient before shading.\nIt can be negative.")));

		//fichiers de sortie
	QLabel* outLabel = new QLabel(tr("image output"));
	outEdit = new QLineEdit;
	paramConvert8B.out = dossier + QString("Geo[I|Ter]\'carte\'/Conversion/Z_Num\'numero\'_DeZoom\'dezoom\'_Geom-Im-\'carte\'_Shade\'parametres\'.tif");
	outEdit->setText(paramConvert8B.out);
	outEdit->setToolTip(conv(tr("8 bit images (full path)\n\'...\' terms indicates\n\'repertoire\' - data directory,\n\'carte\' - depth map reference image number,\n\'dezoom\' - depth map zoom (and \'numero\' - processing step number)\nand \'parametres\' - selected parameters.")));
	outEdit->setMinimumWidth(200);
	outEdit->setMaximumWidth(200);
	
		//anisotropie
	QLabel* anisoLabel = new QLabel(tr("Anisotropy"));
	anisoEdit = new QLineEdit;
	anisoEdit->setText(QVariant(paramConvert8B.anisotropie).toString());
	anisoEdit->setMaximumWidth(50);
	anisoEdit->setMinimumWidth(50);
	anisoEdit->setToolTip(conv(tr("If null, makes isotropic shading (all directions are equivalent). The closer it is to 1, the more directions close to 'the north' have an important role. It is between 0 and 1.")));

		//couleurs hypsométriques
	QLabel* hypsoLabel = new QLabel(conv(tr("Hypsometric colours :")));
	QLabel* hdynLabel = new QLabel(tr("dynamic"));
	hdynEdit = new QLineEdit;
	hdynEdit->setText(QVariant(paramConvert8B.hypsoDyn).toString());
	hdynEdit->setMaximumWidth(50);
	hdynEdit->setMinimumWidth(50);
	hdynEdit->setToolTip(conv(conv(tr("Hypsometric colour dynamic"))));
	QLabel* hsatLabel = new QLabel(tr("saturation"));
	hsatEdit = new QLineEdit;
	hsatEdit->setText(QVariant(paramConvert8B.hypsoSat).toString());
	hsatEdit->setMaximumWidth(50);
	hsatEdit->setMinimumWidth(50);
	hsatEdit->setToolTip(conv(tr("Hypsometric colour saturation")));

		//boîte englobante
	QLabel* boiteLabel = new QLabel(conv(tr("Bounding box :")));
	boiteCheck = new QCheckBox(conv(tr("Use a bounding box")));
	boiteCheck->setChecked(paramConvert8B.doBoite);
	boiteCheck->setToolTip(conv(tr("If selected, only the part of image contained in the bounding box is computed.")));
	boiteArea = new BoiteArea(dossier+cartes->at(0).getRefImg(), cartes->at(0).getDezoom());
	boiteArea->setMinimumSize(150,150);
	paramConvert8B.boite.first = boiteArea->getBoite().first;
	paramConvert8B.boite.second = boiteArea->getBoite().second-boiteArea->getBoite().first;

		//bordure
	QLabel* brdLabel = new QLabel(tr("Edge width"));
	brdBox = new QSpinBox;
	brdBox->setMaximum(10);
	brdBox->setMinimum(0);
	brdBox->setValue(paramConvert8B.bord);
	brdBox->setMaximumWidth(50);
	brdBox->setMinimumWidth(50);
	brdBox->setToolTip(conv(tr("Values close to image edge are often noisy (or even without meaning) ; if they are hight, they can strongly influence the shading. If this parameter is not null, \'Edge width\' applies the minimum value of the image to the parameter value edge width, so that the image edge does not influence shading.")));

	QFormLayout* optionLayout = new QFormLayout;
	optionLayout->addRow(imageCheck);
	optionLayout->addRow(nbdirLabel,nbdirBox);
	optionLayout->addRow(modeombreLabel,modeOmbreBox);
	optionLayout->addRow(fzLabel,fzEdit);
	optionLayout->addRow(outLabel,outEdit);
	optionLayout->addRow(anisoLabel,anisoEdit);
	optionLayout->addRow(hypsoLabel);
	optionLayout->addRow(hdynLabel,hdynEdit);
	optionLayout->addRow(hsatLabel,hsatEdit);
	optionLayout->addRow(boiteLabel);
	optionLayout->addRow(boiteCheck);
	optionLayout->addRow(boiteArea);
	optionLayout->addRow(brdLabel,brdBox);

	optionBox = new QGroupBox;
	optionBox->setLayout(optionLayout);

	//aperçu
	apercuButton = new QToolButton;

	//boutons
	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	calButton = buttonBox->addButton (tr("Compute"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	assistant = help;
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(listImages,0,Qt::AlignHCenter);
	mainLayout->addWidget(groupBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(optionBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(apercuButton,0,Qt::AlignHCenter);
	mainLayout->addSpacing(25);
	mainLayout->addWidget(buttonBox);
	mainLayout->addStretch();
	optionBox->hide();
	boiteArea->hide();

	setLayout(mainLayout);
	setWindowTitle(tr("Depth map conversion to 8 bit images"));
	adjustSize();
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	connect(optionButton, SIGNAL(clicked()), this, SLOT(showOptions()));
	mapper = new QSignalMapper();
		connect(checkVisu, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkVisu, 0);
		connect(checkMask, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkMask, 1);
		connect(checkDequant, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkDequant, 2);
		connect(fzEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(fzEdit, 3);
		connect(outEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(outEdit, 4);
		connect(anisoEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(anisoEdit, 5);
		connect(hdynEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(hdynEdit, 6);
		connect(hsatEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(hsatEdit, 7);
		connect(boiteCheck, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(boiteCheck, 8);
		connect(boiteArea, SIGNAL(changed()), mapper, SLOT(map()));
		mapper->setMapping(boiteArea, 9);
		connect(brdBox, SIGNAL(valueChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(brdBox, 10);
		connect(imageCheck, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(imageCheck, 11);
		connect(nbdirBox, SIGNAL(valueChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(nbdirBox, 12);
		connect(modeOmbreBox, SIGNAL(currentIndexChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(modeOmbreBox, 13);
	connect(mapper, SIGNAL(mapped(int)),this, SLOT(optionChanged(int)));
	connect(listImages, SIGNAL(itemSelectionChanged()),this, SLOT(carteChanged()));	

	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
	optionChanged(1);
	done = true;
}
InterfCartes8B::~InterfCartes8B() {}

void InterfCartes8B::showOptions() {
	if (optionBox->isVisible()) optionBox->hide();
	else optionBox->show();
	paramConvert8B.otherOptions = optionBox->isVisible();
	adjustSize();
	optionChanged();
}

QString InterfCartes8B::checkParametres(int i) {
//vérificiation de la validité des paramètres, enregistrement et aperçu
	bool b = false;
	double res;
	QString t;
	int res2;
	switch (i) {
		case 0 :	//visu
			paramConvert8B.visualiser = checkVisu->isChecked();
			return QString();
		case 1 :	//mask
			paramConvert8B.useMasque = checkMask->isChecked();
			break;
		case 2 :	//dequant
			paramConvert8B.dequantifier = checkDequant->isChecked();
			break;
		case 3 :	//fz
			res = fzEdit->text().toDouble(&b);
			if (!b) return conv(tr("Uncorrect relief coefficient parameter."));
			paramConvert8B.fz = res;
			break;
		case 4 :	//out
			t = outEdit->text();
			paramConvert8B.out = t;
			return QString();
		case 5 :	//anisotropie
			res = anisoEdit->text().toDouble(&b);
			if (!b) return conv(tr("Uncorrect anisotropy parameter."));
			paramConvert8B.anisotropie = res;
			break;
		case 6 :	//hypsodyn
			res = hdynEdit->text().toDouble(&b);
			if (!b) return conv(tr("Uncorrect dynamic of hypsometric colour parameter."));
			paramConvert8B.hypsoDyn = res;
			break;
		case 7 :	//hypsosat
			res = hsatEdit->text().toDouble(&b);
			if (!b) return conv(tr("Uncorrect saturation of hypsometric colour parameter."));
			paramConvert8B.hypsoSat = res;
			break;
		case 8 :	//do boite
			paramConvert8B.doBoite = boiteCheck->isChecked();
			if (boiteCheck->isChecked()) boiteArea->show();
			else boiteArea->hide();
			break;
		case 9 :	//change boite
			paramConvert8B.boite.first = boiteArea->getBoite().first;
			paramConvert8B.boite.second = boiteArea->getBoite().second-boiteArea->getBoite().first;
			break;
		case 10 :	//bord
			res2 = brdBox->text().toDouble(&b);
			if (!b) return conv(tr("Uncorrect image board width parameter."));
			paramConvert8B.bord = res2;
			break;
		case 11 :	//withImg
			paramConvert8B.withImg = imageCheck->isChecked();
			break;
		case 12 :	//nbDir
			paramConvert8B.nbDir = nbdirBox->value();
			break;
		case 13 :	//modeOmbre
			paramConvert8B.modeOmbre = modeOmbreBox->currentIndex();
			break;
	}
	return QString();
}

void InterfCartes8B::optionChanged(int n) {
//vérificiation de la validité des paramètres, enregistrement et aperçu
	cout << tr("Loading preview...").toStdString() << endl;
	//vérification
	QString err = checkParametres(n);
	if (!err.isEmpty()) {
		cout << err.toStdString() << endl;
		return;	//paramètre invalide
	}
	if (n==0 || n==4) {
		cout << "ok\n";
		return; //modification de out ou visu => pas de nouvel aperçu
	}
	//exécution
	if (!paramConvert8B.getCommande(micmacDir, dir, listeCartes->at(0), dir+QString("tempofile.tif"), dir+QString("tempofile.txt"), false))  {
		cout << tr("Fail to compute depth map preview.").toStdString() << endl;
		return;	//paramètre invalide
	}
	//conversion en tif non tuilé
	QString commande = noBlank(applicationPath()) + QString("/lib/tiff2rgba ") + noBlank(dir) + QString("tempofile.tif ") + noBlank(dir) + QString("tempofile2.tif");
cout << commande.toStdString() << endl;
	if (execute(commande)!=0) {
		cout << tr("Fail to convert temporary depth map into untiled tif format.").toStdString() << endl;
		QFile(dir+QString("tempofile.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		return;
	}
	//aperçu
	QImage image(dir+QString("tempofile2.tif"));
	if (image.isNull()) {
		cout << tr("Temporary depth map image is empty.").toStdString() << endl;
		QFile(dir+QString("tempofile.tif")).remove();
		QFile(dir+QString("tempofile2.tif")).remove();
		QFile(dir+QString("tempofile.txt")).remove();
		return;
	}
	image = image.scaled(150,150,Qt::KeepAspectRatio);
	apercuButton->setIconSize(image.size());
	apercuButton->setIcon(QPixmap::fromImage(image));
	apercuButton->adjustSize();
	adjustSize();
	//nettoyage
	QFile(dir+QString("tempofile.tif")).remove();
	QFile(dir+QString("tempofile2.tif")).remove();
	QFile(dir+QString("tempofile.txt")).remove();
	cout << "ok\n";
	return;
}

void InterfCartes8B::carteChanged() {
	if (listImages->selectedItems().size()==0) return;
	int idx0 = 0;
	while (!listImages->item(idx0)->isSelected() && idx0<listImages->count()) idx0++;	
	boiteArea->changeImage(dir+listeCartes->at(idx0).getRefImg());
	paramConvert8B.boite.first = boiteArea->getBoite().first;
	paramConvert8B.boite.second = boiteArea->getBoite().second-boiteArea->getBoite().first;
}

void InterfCartes8B::calcClicked()
{
	for (int i=0; i<10; i++) {
		QString err = checkParametres(i);
		if (!err.isEmpty()) {
			qMessageBox(this, conv(tr("Parameter error.")), err);
			return;
		}
	}

	paramConvert8B.modifImages().clear();
	QList<QListWidgetItem *> l=listImages->selectedItems();
	if (l.count()==0) {
		qMessageBox(this, conv(tr("Parameter error")),conv(tr("No depth maps selected.")));
		return;
	}
	for (QList<QListWidgetItem *>::const_iterator it=l.begin(); it!=l.end(); it++){
		ParamConvert8B::carte8B carte;
		for (QList<ParamConvert8B::carte8B>::const_iterator it2=listeCartes->begin(); it2!=listeCartes->end(); it2++){
			if ((*it)->text()==it2->getCarte16B()) {
				carte = (*it2);
				break;
			}
		}
		paramConvert8B.modifImages().push_back( carte );
	}

	hide();
	accept();
}

void InterfCartes8B::helpClicked() { assistant->showDocumentation(assistant->pageInterfCartes8B); }

const ParamConvert8B& InterfCartes8B::getParam() const { return paramConvert8B; }
bool InterfCartes8B::getDone() const { return done; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////


ParamConvert8B::ParamConvert8B():
	visualiser( false ),
	useMasque( true ),
	dequantifier( true ),
	otherOptions( false ),
	withImg( false ),
	nbDir( 20 ),
	modeOmbre( 0 ),
	fz( 1 ),
	out( QString() ),
	anisotropie( 0.95 ),
	hypsoDyn( 0 ),
	hypsoSat( 0 ),
	doBoite( false ),
	boite( pair<QPoint,QPoint>( QPoint(0,0), QPoint(0,0) ) ),
	bord( 0 ),
	images( QList<carte8B>() ){}
ParamConvert8B::ParamConvert8B(const ParamConvert8B& paramConvert8B) { copie(paramConvert8B); }
ParamConvert8B::~ParamConvert8B() {}

ParamConvert8B& ParamConvert8B::operator=(const ParamConvert8B& paramConvert8B) {
	if (&paramConvert8B!=this) {
		copie(paramConvert8B);
	}
	return *this;
}

void ParamConvert8B::copie(const ParamConvert8B& paramConvert8B) {	
	visualiser = paramConvert8B.visualiser;
	useMasque = paramConvert8B.useMasque;
	dequantifier = paramConvert8B.dequantifier;
	images = paramConvert8B.readImages();
	fz = paramConvert8B.fz;
	out = paramConvert8B.out;
	anisotropie = paramConvert8B.anisotropie;
	hypsoDyn = paramConvert8B.hypsoDyn;
	hypsoSat = paramConvert8B.hypsoSat;
	doBoite = paramConvert8B.doBoite;
	boite = paramConvert8B.boite;
	bord = paramConvert8B.bord;
	otherOptions = paramConvert8B.otherOptions;
	withImg = paramConvert8B.withImg;
	nbDir = paramConvert8B.nbDir;
	modeOmbre = paramConvert8B.modeOmbre;
}

QList<ParamConvert8B::carte8B>& ParamConvert8B::modifImages() {return images;}
const QList<ParamConvert8B::carte8B>& ParamConvert8B::getImages() const {return images;}
const QList<ParamConvert8B::carte8B>& ParamConvert8B::readImages() const {return images;}

QString ParamConvert8B::getOutFile(const QString& dossier, const carte8B& image) {
	//dossier
dossier + QString("Geo[I|Ter]\'carte\' / Conversion / Z_Num\'numero\'_DeZoom\'dezoom\'_Geom-Im-\'carte\'_Shade\'parametres\'.tif");
	QString dir = out.section("/",0,-2) + QString("/");
	dir.replace("[I|Ter]", (image.getCarte16B().left(4)==QString("GeoI"))? "I" : "Ter");
	dir.replace("\'repertoire\'",dossier.section("/",0,-1));
	dir.replace("\'carte\'",image.getNumCarte());
	if (dir.left(2)==QString("./")) dir = dossier + dir.right(dir.count()-2);
	//image
	QString nom = out.section("/",-1,-1);
	nom = nom.section(".",0,-2) + QString(".tif");
	nom.replace("\'carte\'",image.getNumCarte());
	nom.replace("\'dezoom\'",QVariant(image.getDezoom()).toString());
	nom.replace("\'numero\'",QVariant(image.getEtape()).toString());
	//parametres
	QString param;
	if (useMasque) param += QString("_Mask");
	if (dequantifier && image.getEtape()!=7) param += QString("_Dequant");
	if (otherOptions) {
		if (fz!=1.0) param += QString("_Fz%1").arg(fz);
		if (anisotropie!=0.95) param += QString("_Aniso%1").arg(anisotropie);
		if (hypsoDyn!=0) param += QString("_HDyn%1").arg(hypsoDyn);
		if (hypsoSat!=0) param += QString("_HSat%1").arg(hypsoSat);
		if (doBoite) param += QString("_Box%1-%2-%3-%4").arg(boite.first.x()).arg(boite.first.y()).arg(boite.second.x()).arg(boite.second.y());
		if (bord!=0) param += QString("_Brd%1").arg(bord);
		if (withImg) param += QString("_Colour");
		if (nbDir!=20) param += QString("_%1dir").arg(nbDir);
		switch (modeOmbre) {
			case 0 : param += QString("_CielVu");
				break;
			case 1 : param += QString("_Local");
				break;
			case 2 : param += QString("_Mixte");
				break;
			case 3 : param += QString("_Med");
				break;
			case 4 : param += QString("_IgnE");
				break;
		}
	}
	//sortie
	nom.replace("\'parametres\'",param);
	return noBlank(dir)  + nom;
}

bool ParamConvert8B::getCommande(const QString& micmacDir, const QString& dir, const carte8B& image, const QString& outimage, const QString& outstd, bool holdvisu) {	//holdvisu=false permet de ne pas charger la fenêtre de visualisation pour l'aperçu
//retrouve la commande de GrShade à partir des paramètres et l'exécute
	//image à l'échelle
	QString imgRescaled;
	if (otherOptions && withImg) {
		if (image.getDezoom()!=1) {
			imgRescaled = image.getTexture().section(".",0,-2)+QString("sc%1.tif").arg(image.getDezoom());
			QString commande2 = comm(QString("%1bin/ScaleIm %2 %3 Out=%4").arg(noBlank(micmacDir)).arg(noBlank(dir+image.getTexture())).arg(image.getDezoom()).arg(noBlank(dir+imgRescaled)));
			if ( system( commande2.toStdString().c_str() )!=0 )
				return false;
		} else
			imgRescaled = image.getTexture();
	}

	//commande
	QString commande = noBlank(micmacDir) + QString("bin/GrShade ");
	commande += noBlank(dir) + image.getCarte16B();
	commande += QString(" Out=") + noBlank(outimage);
	if (visualiser && holdvisu) commande += QString(" Visu=1");
	if (useMasque) commande += QString(" Mask=") + noBlank(image.getMasque());
	if (dequantifier && image.getEtape()!=7) commande += QString(" Dequant=1");	//l'étape 7 est déjà déquantifiée dans les données
	if (otherOptions) {
		if (fz!=1.0) commande += QString(" FZ=%1").arg(fz);
		if (anisotropie!=0.95) commande += QString(" Anisotropie=%1").arg(anisotropie);
		if (hypsoDyn!=0) commande += QString(" HypsoDyn=%1").arg(hypsoDyn);
		if (hypsoSat!=0) commande += QString(" HypsoSat=%1").arg(hypsoSat);
		if (doBoite) commande += QString(" P0=[%1,%2] Sz=[%3,%4]").arg(boite.first.x()).arg(boite.first.y()).arg(boite.second.x()).arg(boite.second.y());
		if (bord!=0) commande += QString(" Brd=%1").arg(bord);
		if (withImg) commande += QString(" FileCol=../%1").arg(imgRescaled);
		if (nbDir!=20) commande += QString(" NbDir=%1").arg(nbDir);
		switch (modeOmbre) {
			case 0 : commande += QString(" ModeOmbre=CielVu");
				break;
			case 1 : commande += QString(" ModeOmbre=Local");
				break;
			case 2 : commande += QString(" ModeOmbre=Mixte");
				break;
			case 3 : commande += QString(" ModeOmbre=Med");
				break;
			case 4 : commande += QString(" ModeOmbre=IgnE");
				break;
		}
	}
	commande += QString(" >") + noBlank(outstd);
cout << commande.toStdString() << endl;
	#if (defined Q_OS_LINUX || defined Q_WS_MAC)
	if (system (commande.toStdString().c_str() )!=0) {
	#else
	if (execute(comm(commande))!=0) {
	#endif
		if (!imgRescaled.isEmpty()) deleteFile(dir+imgRescaled);
		return false;
	}
	if (!imgRescaled.isEmpty()) deleteFile(dir+imgRescaled);
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////


ParamConvert8B::carte8B::carte8B() : carte16B(QString()), dezoom(1), numCarte(QString()), etape(7), masque(QString()), refImg(QString()) {}
ParamConvert8B::carte8B::carte8B(const QString& c, int dz, QString nc, int e, const QString& m, const QString& ri, const QString& t) : carte16B(c), dezoom(dz), numCarte(nc), etape(e), masque(m), refImg(ri), texture(t) {}
ParamConvert8B::carte8B::carte8B(const ParamConvert8B::carte8B& c) { copie(c); }
ParamConvert8B::carte8B::~carte8B() {}

ParamConvert8B::carte8B& ParamConvert8B::carte8B::operator=(const ParamConvert8B::carte8B& c) {
	if (&c!=this) {
		copie(c);
	}
	return *this;
}

void ParamConvert8B::carte8B::copie(const ParamConvert8B::carte8B& c) {	
	carte16B = c.getCarte16B();
	dezoom = c.getDezoom();
	numCarte = c.getNumCarte();
	etape = c.getEtape();
	masque = c.getMasque();
	refImg = c.getRefImg();
	texture = c.getTexture();
}

const QString& ParamConvert8B::carte8B::getCarte16B() const { return carte16B; }
int ParamConvert8B::carte8B::getDezoom() const { return dezoom; }
QString ParamConvert8B::carte8B::getNumCarte() const { return numCarte; }
int ParamConvert8B::carte8B::getEtape() const { return etape; }
const QString& ParamConvert8B::carte8B::getMasque() const { return masque; }
const QString& ParamConvert8B::carte8B::getRefImg() const { return refImg; }
const QString& ParamConvert8B::carte8B::getTexture() const { return texture; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////


BoiteArea::BoiteArea() : image(QImage()), P1(QPoint(0,0)), P3(QPoint(0,0)), movingPoint(0), dezoom(1), scale(1), precPoint(QPoint(0,0)) { }
BoiteArea::BoiteArea(const QString& img, int dz) : image(QImage(img)), P1(QPoint(image.width()/4,image.height()/4)), P3(QPoint(image.width()*3/4,image.height()*3/4)), movingPoint(0), dezoom(double(1.0)/double(dz)), scale(1), precPoint(QPoint(0,0)) {	//P1(QPoint(0,0)), P3(QPoint(image.width(),image.height())), 
	if (image.isNull()) {
		qMessageBox(this, tr("Read error."), conv(tr("Fail to open image %1.")).arg(img));
		return;
	}
}
BoiteArea::~BoiteArea() {}

pair<QPoint,QPoint> BoiteArea::getBoite() const {
	int x0 = min(P1.x(), P3.x())*dezoom;
	int y0 = min(P1.y(), P3.y())*dezoom;
	int xN = max(P1.x(), P3.x())*dezoom;
	int yN = max(P1.y(), P3.y())*dezoom;
	return pair<QPoint,QPoint>(QPoint(x0,y0),QPoint(xN,yN));
}

void BoiteArea::changeImage(const QString& img) {
	QSize size = image.size();
	image = QImage(img);
	if (image.isNull()) {
		qMessageBox(this, tr("Read error."), conv(tr("")).arg(img));
		return;
	}
	if (image.size()!=size) {
		P1 = QPoint(image.width()/4,image.height()/4);
		P3 = QPoint(image.width()*3/4,image.height()*3/4);
	}	
	update();
}

void BoiteArea::mousePressEvent(QMouseEvent* event) {
	if (!rect ().contains(event->pos())) return;
	precPoint = event->pos();
	QVector<double> V(4,0);
	V[0] = dist(precPoint,P1);
	V[1] = dist(precPoint,P2());
	V[2] = dist(precPoint,P3);
	V[3] = dist(precPoint,P4());
	movingPoint = distance( V.begin() , min_element(V.begin(),V.end()) ) + 1;
}

QPoint BoiteArea::P2() const { return QPoint (P1.x(), P3.y()); }
QPoint BoiteArea::P4() const { return QPoint (P3.x(), P1.y()); }
double BoiteArea::dist (const QPoint& P, const QPoint& PP) const {
	return (P.x()-PP.x()/scale)*(P.x()-PP.x()/scale) + (P.y()-PP.y()/scale)*(P.y()-PP.y()/scale);
}

void BoiteArea::mouseMoveEvent(QMouseEvent* event) {
	if (movingPoint==0) return;
	if (!rect ().contains(event->pos())) return;
	QPoint P(event->pos()-precPoint);
	switch (movingPoint) {
		case 1 : P1 += P*scale;
			break;
		case 2 : P1.setX(P1.x()+P.x()*scale);
			P3.setY(P3.y()+P.y()*scale);
			break;
		case 3 : P3 += P*scale;
			break;
		case 4 : P3.setX(P3.x()+P.x()*scale);
			P1.setY(P1.y()+P.y()*scale);
			break;
	}
	precPoint = event->pos();
	update();
	emit changed();
}

void BoiteArea::mouseReleaseEvent(QMouseEvent*) {
	movingPoint = 0;
	precPoint = QPoint(0,0);
}

void BoiteArea::paintEvent(QPaintEvent*) {	
	//taille du dessin
	QRect rectangleSource(0,0,image.width(),image.height());
	scale = max( double(image.width())/double(width()) , double(image.height())/double(height()) );
	QRect rectangleDestination(0,0,image.width()/scale,image.height()/scale);
	//effacement
	/*QPainter painter0(&image2);
	QBrush brush(Qt::SolidPattern);
	brush.setTextureImage(image);
	painter0.fillRect( QRect(QPoint(0,0),image.size()), brush);
	painter0.end();	*/
	QImage image2 = image;
	//boite	
	QPainter painter1(&image2);
	painter1.setBrush(QBrush(QColor(255,0,0),Qt::Dense3Pattern));
	QPen pen(QColor(255,0,0));
	pen.setWidth(3);
	painter1.setPen(pen);
	painter1.drawRect(P1.x(), P1.y(), P3.x()-P1.x(), P3.y()-P1.y());
	painter1.end();
	//dessin
	QPainter painter(this);
     	painter.drawImage(rectangleDestination, image2, rectangleSource);
	painter.end();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


InterfModele3D::InterfModele3D(QWidget* parent, Assistant* help, ParamMain& param, const QVector<ParamNuages>& choix):
	QDialog( parent ),
	mouse( QPoint(-1,-1) ),
	paramMain( &param ),
	nuages( choix ),
	paramPly( choix.count() ),
	dir( param.getDossier() )
{
	setWindowModality(Qt::ApplicationModal);

	//liste de nuages
	QLabel* plyLabel = new QLabel(conv(tr("Select depth maps to convert :")));
	treeWidget = new QTreeWidget;
	treeWidget->setColumnCount(5);
	treeWidget->setSelectionMode (QAbstractItemView::ExtendedSelection);
	treeWidget->setHeaderLabels (QStringList(conv(tr("Reference images")))
					<<tr("Zoom out")
					<<tr("Step")
					<<tr("Ply file")
					<<tr("Filtering"));
	treeWidget->setPalette(QPalette(QColor(255,255,255)));
	for (int i=0; i<choix.count(); i++) {
		QTreeWidgetItem* twi = new QTreeWidgetItem(QStringList(QDir(dir).relativeFilePath(choix.at(i).getCarte()))<<QVariant(choix.at(i).getDezoom()).toString()<<QVariant(choix.at(i).getEtape()).toString()<<QDir(dir).relativeFilePath(choix.at(i).getFichierPly())<<QString());
		treeWidget->addTopLevelItem(twi);
	}
	resizeTreeWidget();

	//autres options
		//intervalle d'échantillonnage
	QLabel* intervLabel = new QLabel(conv(tr("Subscaling interval")));
	intervBox = new QSpinBox;
	intervBox->setMaximum(100);
	intervBox->setMinimum(1);
	intervBox->setValue(paramPly.getEchantInterval());
	intervBox->setMaximumWidth(50);
	intervBox->setMinimumWidth(50);
	intervBox->setToolTip(conv(tr("Allows to compute only one point per interval without modifying cloud scale.")));

		//résultats au format binaire
	checkBinaire = new QCheckBox(conv(tr("Results in binary mode")));
	checkBinaire->setToolTip(conv(tr("If selected, resulting files are in binary mode (ply file is readable by MeshLab), otherwise they are in text mode.")));
	checkBinaire->setChecked(paramPly.getBinaire());

		//nuage ply
	checkPly = new QCheckBox(tr("Ply file"));
	checkPly->setToolTip(conv(tr("If checked, creates a ply file (if it is in binary mode, it is readable by MeshLab).")));
	checkPly->setChecked(paramPly.getDoPly());

		//nuage xyz
	checkXyz = new QCheckBox(tr("xyz file"));
	checkXyz->setToolTip(conv(tr("If checked, creates a file in xyz format.")));
	checkXyz->setChecked(paramPly.getDoXyz());

		//exagération du relief
	QLabel* reliefLabel = new QLabel(conv(tr("Relief")));
	reliefEdit = new QLineEdit;
	reliefEdit->setText(QVariant(paramPly.getExagZ()).toString());
	reliefEdit->setMaximumWidth(50);
	reliefEdit->setMinimumWidth(50);
	reliefEdit->setToolTip(conv(tr("If >1, exagerates the relief, otherwise flatten it.")));

		//dynamique (luminosité)
	QLabel* dynLabel = new QLabel(conv(tr("Image dynamic")));
	dynEdit = new QLineEdit;
	dynEdit->setText(QVariant(paramPly.getDyn()).toString());
	dynEdit->setMaximumWidth(50);
	dynEdit->setMinimumWidth(50);
	dynEdit->setToolTip(conv(tr("If >1, increases texture image light, otherwise darken it.")));

		//boîte englobante
	QLabel* boiteLabel = new QLabel(conv(tr("Bounding box :")));
	boiteCheck = new QCheckBox(conv(tr("Use a bounding box")));
	boiteCheck->setChecked(paramPly.getDoBoite());
	boiteCheck->setToolTip(conv(tr("If selected, only the part of image contained in the bounding box is computed.")));
	if (!QFile(paramMain->convertTifName2Couleur(nuages.at(0).getCarte())).exists()) {
		QString commande = noBlank(applicationPath()) + QString("/lib/tiff2rgba ") + noBlank(nuages.at(0).getCarte()) + QString(" ") + noBlank(paramMain->convertTifName2Couleur(nuages.at(0).getCarte()));
		if (execute(commande)!=0)
			qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to convert image %1 to untiled tif format.")).arg(dir+nuages.at(0).getCarte()));
	}
	boiteArea = new BoiteArea(paramMain->convertTifName2Couleur(nuages.at(0).getCarte()), 1);
	boiteArea->setMinimumSize(150,150);
	paramPly.setBoite(boiteArea->getBoite());
	
		//filtrage automatique
	checkFiltr = new QCheckBox(conv(tr("Noisy point filtering")));
	checkFiltr->setToolTip(conv(tr("Automatically filters noisy points that come from scene discontinuities.\nIf selected, manual filtering is not taken into account.")));
	checkFiltr->setChecked(paramPly.getDoFiltrage());
	
		//fusion automatique
	checkFusion = new QCheckBox(tr("Merge clouds"));
	checkFusion->setToolTip(conv(tr("Simplifies duplicated points and creates a single cloud per selected step.")));
	checkFusion->setChecked(paramPly.getDoFusion());
	
		//égalisation radiométrique automatique
	checkRadiomEq = new QCheckBox(conv(tr("Radiometric equalization")));
	checkRadiomEq->setToolTip(conv(tr("Makes a radiometric equalization before ply conversion for the images used to texture the cloud. Clouds must be merged first.")));
	checkRadiomEq->setChecked(paramPly.getDoEgalRadiom());

	QFormLayout* optionLayout = new QFormLayout;
	optionLayout->addRow(intervLabel,intervBox);
	optionLayout->addRow(checkBinaire);
	optionLayout->addRow(checkPly);
	optionLayout->addRow(checkXyz);
	optionLayout->addRow(reliefLabel,reliefEdit);
	optionLayout->addRow(dynLabel,dynEdit);
	optionLayout->addRow(boiteLabel);
	optionLayout->addRow(boiteCheck);
	optionLayout->addRow(boiteArea);
	optionLayout->addRow(checkFiltr);
	//optionLayout->addRow(checkFusion);
	//optionLayout->addRow(checkRadiomEq);
	checkRadiomEq->hide();
	boiteArea->hide();

	QGroupBox* optionBox = new QGroupBox;
	optionBox->setLayout(optionLayout);

	//boutons
	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	calButton = buttonBox->addButton (tr("Compute"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	assistant = help;
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(plyLabel,0,Qt::AlignHCenter);
	mainLayout->addWidget(treeWidget,0,Qt::AlignHCenter);
	mainLayout->addWidget(optionBox,0,Qt::AlignHCenter);
	mainLayout->addSpacing(25);
	mainLayout->addWidget(buttonBox);
	mainLayout->addStretch();

	setLayout(mainLayout);
	setWindowTitle(tr("Depth map conversion into ply point clouds"));
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	changeImgAct = new QAction(tr("&Other texture image"), this);
	connect(changeImgAct, SIGNAL(triggered()), this, SLOT(imgChoose()));
	savePlyAct = new QAction(conv(tr("&Other saving directory")), this);
	connect(savePlyAct, SIGNAL(triggered()), this, SLOT(dirPlyChoose()));
	maskFiltrPlyAct = new QAction(conv(tr("&Noisy point filtering mask")), this);
	maskFiltrPlyAct->setToolTip(conv(tr("Allows to draw a mask to filter noisy points that come from scene discontinuities. It is not taken into account if automatic filtering is checked.")));
	connect(maskFiltrPlyAct, SIGNAL(triggered()), this, SLOT(drawMaskFiltr()));

	mapper = new QSignalMapper();
		connect(intervBox, SIGNAL(valueChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(intervBox, 0);
		connect(checkBinaire, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkBinaire, 1);
		connect(checkPly, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkPly, 2);
		connect(checkXyz, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkXyz, 3);
		connect(reliefEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(reliefEdit, 4);
		connect(dynEdit, SIGNAL(textChanged(QString)), mapper, SLOT(map()));
		mapper->setMapping(dynEdit, 5);
		connect(boiteCheck, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(boiteCheck, 6);
		connect(boiteArea, SIGNAL(changed()), mapper, SLOT(map()));
		mapper->setMapping(boiteArea, 7);
		connect(checkFiltr, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkFiltr, 8);
		connect(checkFusion, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkFusion, 9);
		connect(checkRadiomEq, SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(checkRadiomEq, 10);
	connect(mapper, SIGNAL(mapped(int)),this, SLOT(optionChanged(int)));
	connect(treeWidget, SIGNAL(itemSelectionChanged()),this, SLOT(nuageChanged()));	

	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
}

InterfModele3D::~InterfModele3D() {}

void InterfModele3D::showEvent(QShowEvent*) { paramPly.setNuages( QVector<bool>(nuages.count(),false) ); }

void InterfModele3D::resizeTreeWidget() {	
	QFontMetrics metrics = QFontMetrics(treeWidget->font());
	int h = 0;
	for (int k=0; k<treeWidget->columnCount(); k++) {
		treeWidget->resizeColumnToContents(k);
		QString text = treeWidget->headerItem()->text(k);
		for (int i=0; i<treeWidget->topLevelItemCount(); i++) {
			text += QString("\n") + treeWidget->topLevelItem(i)->text(k);
		}
		QRect maxr(0,0,QApplication::desktop()->availableGeometry().width()/2,QApplication::desktop()->availableGeometry().height()/2);
		QRect r = metrics.boundingRect(maxr, Qt::AlignLeft|Qt::AlignVCenter, text);
		if (r.height()>h) h = r.height();
	}
	treeWidget->setFixedHeight(min(h+40,200));	//ascenseur + en-têtes de treeWidget
	treeWidget->setFixedWidth(treeWidget->columnWidth(0)+treeWidget->columnWidth(1)+treeWidget->columnWidth(2)+treeWidget->columnWidth(3));
}

void InterfModele3D::contextMenuEvent(QContextMenuEvent *event) {	
	if (treeWidget->geometry().contains(treeWidget->parentWidget()->mapFrom(this,event->pos()))) {
		//mouse = treeWidget->parentWidget()->mapFrom(this,event->pos());
		mouse = treeWidget->mapFrom(treeWidget->parentWidget(),event->pos()) - QPoint(0,treeWidget->header()->height());
		QMenu menu(this);
		menu.addAction(changeImgAct);
		menu.addAction(savePlyAct);
		menu.addAction(maskFiltrPlyAct);
		menu.exec(event->globalPos());
	} 
}

void InterfModele3D::imgChoose() {
	//image pour la texture
	FileDialog fileDialog(this, tr("Image loading"), dir, tr("Tif files (*.tif)") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec())
		fileNames = fileDialog.selectedFiles();
	else return;
  	if (fileNames.size()!=1)
		return;
	QString texture = QDir(dir).relativeFilePath(fileNames.at(0));	
	//nuage à modifier
	int idx = treeWidget->indexOfTopLevelItem(treeWidget->itemAt(mouse));
	if (idx==-1) {
		qMessageBox(this, tr("Selection error"), conv(tr("The selected line does not match any depth map")));
		return;
	}
	QString carte = nuages.at(idx).getCarte();
	//modification pour chaque étape
	for (int i=0; i<nuages.count(); i++) {
		if (nuages.at(i).getCarte()==carte) {
			nuages[i].setCarte(fileNames.at(0));
			treeWidget->topLevelItem(i)->setText(0,texture);
		}
	}
	if (!QFile(paramMain->convertTifName2Couleur(dir+texture)).exists()) {
		QString commande = noBlank(applicationPath()) + QString("/lib/tiff2rgba ") + noBlank(dir+texture) + QString(" ") + noBlank(dir+paramMain->convertTifName2Couleur(texture));
		if (execute(commande)!=0)
			qMessageBox(this, tr("Execution error"), conv(tr("Fail to convert image %1 to untiled tif format.")).arg(dir+texture));
	}
	if (treeWidget->topLevelItem(idx)->isSelected()) nuageChanged();	//pour remettre à jour l'image de BoiteArea
}

void InterfModele3D::dirPlyChoose() {
	//répertoire du nuage ply
	FileDialog fileDialog(this, tr("Ply point cloud saving"), dir);
	fileDialog.setFilter(QDir::Dirs);
	fileDialog.setFileMode(QFileDialog::Directory);
	fileDialog.setOption(QFileDialog::ShowDirsOnly);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList dirNames;
	if (fileDialog.exec())
		dirNames = fileDialog.selectedFiles();
	else return;
  	if (dirNames.size()!=1)
		return;
	QString saveDir = QDir(dir).relativeFilePath(dirNames.at(0)) + QString("/");
	cout << saveDir.toStdString() << endl;		//laisser sinon pb de pointeur
	//nuage à modifier
	int idx = treeWidget->indexOfTopLevelItem(treeWidget->itemAt(mouse));
	if (idx==-1) {
		qMessageBox(this, tr("Selection error"), conv(tr("The selected line does not match any depth map")));
		return;
	}
	QString carte = nuages.at(idx).getCarte();
	//modification pour chaque étape
	for (int i=0; i<nuages.count(); i++) {
		if (nuages.at(i).getCarte()==carte) {
			QString fichierPly = nuages.at(i).getFichierPly().section("/",-1,-1);
			nuages[i].setFichierPly(dirNames.at(0)+QString("/")+fichierPly);
			treeWidget->topLevelItem(i)->setText(3,saveDir+fichierPly);
		}
	}	
}

void InterfModele3D::drawMaskFiltr() {
//dessin du masque de filtrage des points bruités
	//nuage à modifier
	int idx = treeWidget->indexOfTopLevelItem(treeWidget->itemAt(mouse));
	if (idx==-1) {
		qMessageBox(this, tr("Selection error"), conv(tr("The selected line does not match any depth map")));
		return;
	}
	QString numCarte = nuages.at(idx).getNumCarte();

	//choix du masque de départ
	FileDialog fileDialog(this, tr("Open a mask"), dir, tr("Mask (*.tif);;Mask (*.tiff)") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) {
		fileNames = fileDialog.selectedFiles();
	} else return;
  	if (fileNames.size()>1) return;
	else if (fileNames.size()==0) {
		//désactivation du filtrage
		for (int i=0; i<nuages.count(); i++) {	//affichage du masque dans le tableau
			if (nuages.at(i).getNumCarte()==numCarte)
				treeWidget->topLevelItem(i)->setText(4,QString());
		}
	}

	QString masqueInit = fileNames.at(0);
	QString masqueInit2 = imgNontuilee(masqueInit);
	QString err = MasqueWidget::convert2Rgba(masqueInit, true, masqueInit2);
	if (!err.isEmpty()) {
		qMessageBox(this, conv(tr("Execution error")), err);
		return;
	}
	deleteFile(applicationPath()+QString("/masquetempo.tif"));
	QFile(masqueInit).copy(applicationPath()+QString("/masquetempo.tif"));
	deleteFile(imgNontuilee(applicationPath()+QString("/masquetempo.tif")));
	QFile(masqueInit2).rename(imgNontuilee(applicationPath()+QString("/masquetempo.tif")));
	deleteFile(masqueInit2);
	
	//carte de corrélation
	QString correlFile = dir + QString("Geo%1%2/Correl_Geom-Im-%2_Num_6.tif").arg((nuages.at(idx).getParamMasque().getRepere())? QString("I") : QString("Ter")).arg(numCarte);
	QString correlFile2 = imgNontuilee(correlFile);
	err = MasqueWidget::convert2Rgba(correlFile, false, correlFile2);
	if (!err.isEmpty()) {
		qMessageBox(this, conv(tr("Execution error")), err);
		return;
	}

	//mise à l'échelle du masque (cas du TA où le masque est 8* trop petit pour l'image de corrélation ET pour Nuage2Ply)
	QSize corrSize = QImage(correlFile2).size();
	QImage imgMasque = QImage(imgNontuilee(applicationPath()+QString("/masquetempo.tif")));
	QSize maskSize = imgMasque.size();
	bool rescale = false;
	if (corrSize/8==maskSize) {
		rescale = true;
		//masque non tuilé
		imgMasque = imgMasque.scaled(corrSize);
		imgMasque.save(imgNontuilee(applicationPath()+QString("/masquetempo.tif")));
		//masque tuilé
		QString commande = comm(QString("%1bin/ScaleIm %2/masquetempo.tif 0.125").arg(noBlank(paramMain->getMicmacDir())).arg(noBlank(applicationPath())));
		if (system( commande.toStdString().c_str() )!=0) {
			qMessageBox(this, tr("Execution error"), conv(tr("Fail to rescale mask %1.")).arg(masqueInit));
			return;
		}
		deleteFile(applicationPath()+QString("/masquetempo.tif"));
		QFile(applicationPath()+QString("/masquetempo_Scaled.tif")).rename(applicationPath()+QString("/masquetempo.tif"));
	}

	//ouverture de la fenêtre de dessin
	PaintInterfPlan* paintInterf = new PaintInterfPlan(correlFile2, paramMain, assistant, this, true, applicationPath()+QString("/masquetempo.tif"));
	if (!paintInterf->getDone()) {
		delete paintInterf;
		return;
	}
	paintInterf->show();
	if (paintInterf->exec() != QDialog::Accepted) {
		delete paintInterf;
		return;
	}
	paintInterf->getMaskImg();
	bool masqueIsEmpty = paintInterf->masqueIsEmpty();
	if (rescale) qMessageBox(this, tr("Warning"), conv(tr("Mask %1 rescaled.\nDo not replace micmac mask when saving it.")).arg(masqueInit));
	delete paintInterf;

	//enregistrement
	QString masqueFiltre;
	if (!masqueIsEmpty || rescale) {
		//fenêtre de sauvegarde
		masqueFiltre = FileDialog::getSaveFileName(this, tr("Save mask"), dir+QString("Geo%1%2/").arg((nuages.at(idx).getParamMasque().getRepere())? QString("I") : QString("Ter")).arg(numCarte), conv(tr("Image (*.tif)")));
		if (masqueFiltre.isEmpty()) return;
		if (masqueFiltre.contains(".")) masqueFiltre = masqueFiltre.section(".",0,-2);
		masqueFiltre += QString(".tif");
		masqueFiltre = QDir(masqueFiltre.section("/",0,-2)).absoluteFilePath(masqueFiltre);

		//suppression des fichiers inutiles
		deleteFile(masqueFiltre);
		QFile(applicationPath()+QString("/masquetempo.tif")).rename(masqueFiltre);
	} else {
		masqueFiltre = masqueInit;
		QFile(applicationPath()+QString("/masquetempo.tif")).remove();
	}
	QFile(correlFile2).remove();
	QFile(imgNontuilee(applicationPath()+QString("/masquetempo.tif"))).remove();

		//masque non tuilé pour mise à l'échelle
	err = MasqueWidget::convert2Rgba(masqueFiltre, false);
	if (!err.isEmpty()) {
		qMessageBox(this, conv(tr("Execution error")), err);
		return;
	}

	paramPly.modifMasques().push_back(pair<QString,QString>(numCarte,masqueFiltre));

	masqueFiltre = QDir(dir).relativeFilePath(masqueFiltre);
	if (masqueFiltre.left(2)==QString("./")) masqueFiltre = masqueFiltre.right(masqueFiltre.count()-2);

		//affichage du masque dans le tableau
	for (int i=0; i<nuages.count(); i++) {
		if (nuages.at(i).getNumCarte()==numCarte) treeWidget->topLevelItem(i)->setText(4,masqueFiltre);
		nuages[i].updateFileName(paramPly);
		treeWidget->topLevelItem(i)->setText(3,QDir(dir).relativeFilePath(nuages.at(i).getFichierPly()));
	}
	resizeTreeWidget();
}

void InterfModele3D::nuageChanged() {
	if (treeWidget->selectedItems().size()==0) return;
	int idx0 = 0;
	while (!treeWidget->topLevelItem(idx0)->isSelected() && idx0<treeWidget->topLevelItemCount()) idx0++;	
	boiteArea->changeImage(paramMain->convertTifName2Couleur(dir+treeWidget->topLevelItem(idx0)->text(0)));
	paramPly.setBoite(boiteArea->getBoite());
}

void InterfModele3D::optionChanged(int n) {
//vérificiation de la validité des paramètres, enregistrement, modification du nom du résultat
	bool ok;
	switch (n) {
		case 0 :	//intervBox
			paramPly.setEchantInterval(intervBox->value());			
			break;
		case 1 :	//checkBinaire
			paramPly.setBinaire(checkBinaire->isChecked());	
			break;
		case 2 :	//checkPly
			paramPly.setDoPly(checkPly->isChecked());	
			break;
		case 3 :	//checkXyz
			paramPly.setDoXyz(checkXyz->isChecked());	
			break;
		case 4 :	//reliefEdit
			if (reliefEdit->text().isEmpty()) return;
			paramPly.setExagZ(reliefEdit->text().toDouble(&ok));	
			if (!ok) {
				qMessageBox(this, conv(tr("Parameter error")), conv(tr("Uncorrect relief parameter.")));
				return;
			}
			break;
		case 5 :	//dynEdit
			if (dynEdit->text().isEmpty()) return;
			paramPly.setDyn(dynEdit->text().toDouble(&ok));	
			if (!ok) {
				qMessageBox(this, conv(tr("Parameter error")), conv(tr("Uncorrect image dynamic parameter.")));
				return;
			}
			break;
		case 6 :	//boiteCheck
			paramPly.setDoBoite(boiteCheck->isChecked());	
			if (boiteCheck->isChecked()) boiteArea->show();
			else boiteArea->hide();
			break;
		case 7 :	//boiteArea
			paramPly.setBoite(boiteArea->getBoite());	
			return;
		case 8 :	//checkFiltr
			paramPly.setDoFiltrage(checkFiltr->isChecked());	
			return;
		case 9 :	//checkFusion
			paramPly.setDoFusion(checkFusion->isChecked());
			if (!checkFusion->isChecked()) {
				checkRadiomEq->setChecked(false);
				checkRadiomEq->hide();
			} else
				checkRadiomEq->show();
			return;
		case 10 :	//checkRadiomEq
			paramPly.setDoEgalRadiom(checkRadiomEq->isChecked());
			return;
	}
	for (int i=0; i<nuages.count(); i++) {
		nuages[i].updateFileName(paramPly);
		treeWidget->topLevelItem(i)->setText(3,QDir(dir).relativeFilePath(nuages.at(i).getFichierPly()));
	}
}

void InterfModele3D::calcClicked() {
	//nuages à calculer
	QList<QTreeWidgetItem *> l = treeWidget->selectedItems();
	if (l.count()==0) {
		reject();
		return;
	}
	for (int i=0; i<l.count(); i++)
		paramPly.modifNuages()[treeWidget->indexOfTopLevelItem(l.at(i))] = true;
	//filtrage et fusion
	if (checkFiltr->isChecked()) paramPly.modifMasques().clear();
	//vérification
	if (reliefEdit->text().isEmpty()) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("Missing relief parameter.")));
		return;
	}
	if (dynEdit->text().isEmpty()) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("Missing dynamic parameter.")));
		return;
	}
	
	hide();
	accept();
}

void InterfModele3D::chercheOrtho() {
	for (int i=0; i<nuages.count(); i++) {
		if (!nuages.at(i).getParamMasque().getOrthoCalculee()) continue;
		QString numCarte = paramMain->getNumImage( nuages.at(i).getParamMasque().getImageDeReference(), 0, false );
		nuages[i].setCarte(paramMain->getDossier()+QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numCarte));
		treeWidget->topLevelItem(i)->setText(0, QString("ORTHO%1/Ortho-NonEg-Test-Redr.tif").arg(numCarte));
	}
}

void InterfModele3D::helpClicked() { assistant->showDocumentation(assistant->pageInterfModeles3D); }

const QVector<ParamNuages>& InterfModele3D::getModifications() const { return nuages; }
const ParamPly& InterfModele3D::getParamPly() const { return paramPly; }

/////////////////////////////////////////////////////////////////////////////////////////////////


ParamNuages::ParamNuages():
	paramMasque( 0 ),
	carte( QString() ),
	dezoom( 1 ),
	numCarte( QString() ),
	etape( 7 ),
	fichierPly( QString() ){}
ParamNuages::ParamNuages(const ParamNuages& paramNuages) { copie(paramNuages); }
ParamNuages::~ParamNuages() {}

ParamNuages& ParamNuages::operator=(const ParamNuages& paramNuages) {
	if (&paramNuages!=this) {
		copie(paramNuages);
	}
	return *this;
}

void ParamNuages::copie(const ParamNuages& paramNuages) {	
	carte = paramNuages.getCarte();
	dezoom = paramNuages.getDezoom();
	numCarte = paramNuages.getNumCarte();
	etape = paramNuages.getEtape();
	fichierPly = paramNuages.getFichierPly();
	fichierXml = paramNuages.getFichierXml();
	paramMasque = &paramNuages.getParamMasque();
}

const CarteDeProfondeur& ParamNuages::getParamMasque() const { return *paramMasque; }
const QString& ParamNuages::getCarte() const { return carte; }
int ParamNuages::getDezoom() const { return dezoom; }
QString ParamNuages::getNumCarte() const { return numCarte; }
int ParamNuages::getEtape() const { return etape; }
const QString& ParamNuages::getFichierPly() const { return fichierPly; }
const QString& ParamNuages::getFichierXml() const { return fichierXml; }

void ParamNuages::setParamMasque(const CarteDeProfondeur* p) { paramMasque = p; }
void ParamNuages::setCarte(const QString& c) { carte = c; }
void ParamNuages::setDezoom(int d) { dezoom = d; }
void ParamNuages::setNumCarte(QString n) { numCarte = n; }
void ParamNuages::setEtape(int i) { etape = i; }
void ParamNuages::setFichierPly(const QString& s) { fichierPly = s; }

void ParamNuages::calcFileName(const QString& dossier) {
	fichierPly = dossier + QString("Geo%1%2/Conversion/NuageImProf_Geom-Im-%2_Etape_%3.ply").arg(paramMasque->getRepere()? QString("I") : QString("Ter")).arg(numCarte).arg(etape);
	fichierXml = dossier + QString("Geo%1%2/NuageImProf_Geom-Im-%2_Etape_%3.xml").arg(paramMasque->getRepere()? QString("I") : QString("Ter")).arg(numCarte).arg(etape);
}
void ParamNuages::updateFileName(const ParamPly& paramPly) {
//modifie les noms des fichiers en sorties en fonction du paramétrage (évite d'écraser les fichiers)
	fichierPly = fichierPly.section("/",0,-2) + QString("/NuageImProf_Geom-Im-%1_Etape_%2").arg(numCarte).arg(etape);	//permet de conserver le dossier mais pas l'échelle
	if (paramPly.getEchantInterval()!=1) fichierPly += QString("_scale%1").arg(paramPly.getEchantInterval());
	if (paramPly.getDoBoite()) fichierPly += QString("_box");
	if (!paramPly.getBinaire()) fichierPly += QString("_txt");
	if (paramPly.getExagZ()!=1) fichierPly += QString("_relief%1").arg(paramPly.getExagZ());
	if (paramPly.getDyn()!=1) fichierPly += QString("_dyn%1").arg(paramPly.getDyn());
	if (!paramPly.getMasques().isEmpty()) {
		for (int i=0; i<paramPly.getMasques().count(); i++) {
			if (paramPly.getMasques().at(i).first==numCarte) {
				fichierPly += QString("_filtreMan");
				break;
			}
		}
	}
	fichierPly += QString(".ply");
}

QString ParamNuages::commandeFiltrage(QString& masqueFiltre) const {
	//récupération du nom du masque
	masqueFiltre = fichierXml.section("/",0,-2) + QString("/Masq_Geom-Im-%1_DeZoom1_filtre.tif").arg(numCarte);
	return applicationPath() + QString("/lib/filtrageNuage %1 7 %2").arg(noBlank(fichierXml).section("/",0,-2)).arg(noBlank(carte));
}

QString ParamNuages::commandePly(QString& commandeNuage2Ply, const QString& micmacDir, const ParamPly& paramPly, const QString& masqueFiltre, const ParamMain& paramMain) const {
	deleteFile(fichierXml.section("/",0,-2)+QString("/tempo.tif"));
	commandeNuage2Ply = QString();

	double echelle = dezoom;
	if (!paramMasque->getRepere() && carte==paramMasque->getImageSaisie(paramMain)) echelle /= 8;
	QString TArescaled = carte;
	if (echelle<1) {
		TArescaled = paramMasque->getImageSaisie(paramMain).section(".",0,-2)+QString("_rescaled.tif");
		deleteFile(TArescaled);
		QString commandeScale = micmacDir + QString("bin/ScaleIm %1 %2 Out=%3").arg(noBlank(paramMasque->getImageSaisie(paramMain))).arg(echelle).arg(TArescaled);
		if (execute(commandeScale)!=0) return conv(QObject::tr("Fail to rescale IM."));
	}

	QString commande =  QString("%1bin/Nuage2Ply %2 Attr=%3 RatioAttrCarte=%4").arg(noBlank(micmacDir)).arg(noBlank(fichierXml)).arg(noBlank(TArescaled)).arg(max(int(echelle),1));
	
	commande += QString(" Scale=%1 Out=%2 Bin=%3 DoPly=%4 DoXYZ=%5 Dyn=%6 ExagZ=%7").arg(paramPly.getEchantInterval()).arg(fichierPly).arg(paramPly.getBinaire()).arg(paramPly.getDoPly()).arg(paramPly.getDoXyz()).arg(paramPly.getDyn()).arg(paramPly.getExagZ());

	if (paramPly.getDoBoite())
		commande += QString(" P0=[%1,%2] Sz=[%3,%4]").arg(paramPly.getBoite().first.x()).arg(paramPly.getBoite().first.y()).arg(paramPly.getBoite().second.x()).arg(paramPly.getBoite().second.y());

	if (!masqueFiltre.isEmpty()) {
		if (dezoom==1) commande += QString(" Mask=%1").arg(noBlank(masqueFiltre));
		else {
			//mise à l'échelle du masque
			//QImage masque(imgNontuilee(masqueFiltre));
			//masque = masque.scaled(masque.size()/dezoom);
			QString masqueRescaled = masqueFiltre.section(".",0,-2) + QString("tempo.tif");
			deleteFile(masqueRescaled);
			//masque.save(masqueRescaled);			
			QString commandeScale = micmacDir + QString("bin/ScaleIm %1 %2 Out=%3").arg(noBlank(masqueFiltre)).arg(dezoom).arg(masqueRescaled);
			if (execute(commandeScale)!=0) return conv(QObject::tr("Fail to rescale filtered mask."));
			commande += QString(" Mask=%1").arg(noBlank(masqueRescaled));
		}
	}

	else if (paramPly.getMasques().count()!=0) {
		for (int i=0; i<paramPly.getMasques().count(); i++) {
			if (paramPly.getMasques().at(i).first==numCarte) {
				if (dezoom==1) commande += QString(" Mask=%1").arg(noBlank(paramPly.getMasques().at(i).second));
				else {
					//mise à l'échelle du masque
					QImage masque(imgNontuilee(paramPly.getMasques().at(i).second));
					masque = masque.scaled(masque.size()/dezoom);
					QString masqueRescaled = paramPly.getMasques().at(i).second.section(".",0,-2) + QString("tempo.tif");
					deleteFile(masqueRescaled);
					masque.save(masqueRescaled);
					commande += QString(" Mask=%1").arg(noBlank(masqueRescaled));
				}
				break;
			}
		}
	}
	commandeNuage2Ply = comm(commande);
	return QString();
}

/*QString ParamNuages::commandePly(const QString& micmacDir, const ParamPly& paramPly) const {
	QString image(fichierXml.section("/",0,-2)+QString("/tempo%1.tif").arg(etape));
	QString commande =  QString("%1bin/Nuage2Ply %2 Attr=%3 Scale=%4").arg(noBlank(micmacDir)).arg(noBlank(fichierXml)).arg(noBlank(image)).arg(paramPly.getEchantInterval());	//Attr = l'image couleur
	if (paramPly.getMasques().count()!=0) {
		for (int i=0; i<paramPly.getMasques().count(); i++) {
			if (paramPly.getMasques().at(i).first==numCarte) {
				if (dezoom==1) commande += QString(" Mask=%1").arg(noBlank(paramPly.getMasques().at(i).second));
				else {
					//mise à l'échelle du masque
					QImage masque(imgNontuilee(paramPly.getMasques().at(i).second));
					masque = masque.scaled(masque.size()/dezoom);
					QString masqueRescaled = paramPly.getMasques().at(i).second.section(".",0,-2) + QString("tempo.tif");
					deleteFile(masqueRescaled);
					masque.save(masqueRescaled);
					commande += QString(" Mask=%1").arg(noBlank(masqueRescaled));
				}
				break;
			}
		}
	}
	return commande;
}*/

bool ParamNuages::commandeMv() const {
	bool b = QFile(fichierXml.section(".",0,-2)+QString(".ply")).rename(fichierPly);
	return b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////


ParamPly::ParamPly():
	echantInterval( 1 ),
	nuages( QVector<bool>() ),
	doFiltrage( false ),
	doFusion( false ),
	doEgalRadiom( false ),
	masques( QList<pair<QString,QString> >() ),
	doBoite( false ),
	boite( pair<QPoint,QPoint>( QPoint(0,0), QPoint(0,0) ) ),
	#if (defined Q_WS_WIN)
		binaire(false),
	#else
		binaire(true),
	#endif
	doPly( true ),
	doXyz( false ),
	exagZ( 1 ),
	dyn( 1 )
{}
ParamPly::ParamPly(int nbNuages):
	echantInterval( 1 ),
	nuages( QVector<bool>( nbNuages, false ) ),
	doFiltrage( false ),
	doFusion( false ),
	doEgalRadiom( false ),
	masques( QList<pair<QString,QString> >() ),
	doBoite( false ),
	boite( pair<QPoint,QPoint>( QPoint(0,0), QPoint(0,0) ) ),
	#if (defined Q_WS_WIN)
		binaire(false),
	#else
		binaire(true),
	#endif
	doPly( true ),
	doXyz( false ),
	exagZ( 1 ),
	dyn( 1 )
{}
ParamPly::ParamPly(const ParamPly& paramPly) { copie(paramPly); }
/*ParamPly::ParamPly(int e, const QVector<bool>& n, bool d, bool doF, bool doE, const QList<pair<QString,QString> >& m, bool doB, const Pt2di& P, const Pt2di& S, bool b, bool doP, bool doX, double ex, double dy) : echantInterval(e), nuages(n), doFiltrage(d), doFusion(doF), doEgalRadiom(doE), masques(m), doBoite(doB), P0(P), Sz(S), binaire(b), doPly(doP), doXyz(doX), exagZ(ex), dyn(dy) {}*/
ParamPly::~ParamPly() {}

ParamPly& ParamPly::operator=(const ParamPly& paramPly) {
	if (this!=&paramPly) copie(paramPly);
	return *this;
}

void ParamPly::copie(const ParamPly& paramPly) {
	echantInterval = paramPly.getEchantInterval();
	nuages = paramPly.getNuages();
	doFiltrage = paramPly.getDoFiltrage();
	doFusion = paramPly.getDoFusion();
	doEgalRadiom = paramPly.getDoEgalRadiom();
	masques = paramPly.getMasques();
	doBoite = paramPly.getDoBoite();
	boite = paramPly.getBoite();
	binaire = paramPly.getBinaire();
	doPly = paramPly.getDoPly();
	doXyz = paramPly.getDoXyz();
	exagZ = paramPly.getExagZ();
	dyn = paramPly.getDyn();
}

int ParamPly::getEchantInterval() const { return echantInterval; }
const QVector<bool>& ParamPly::getNuages() const { return nuages; }
bool ParamPly::getDoFiltrage() const { return doFiltrage; }
bool ParamPly::getDoFusion() const { return doFusion; }
bool ParamPly::getDoEgalRadiom() const { return doEgalRadiom; }
const QList<pair<QString,QString> >& ParamPly::getMasques() const { return masques; }
bool ParamPly::getDoBoite() const { return doBoite; }
const pair<QPoint,QPoint>& ParamPly::getBoite() const { return boite; }
bool ParamPly::getBinaire() const { return binaire; }
bool ParamPly::getDoPly() const { return doPly; }
bool ParamPly::getDoXyz() const { return doXyz; }
double ParamPly::getExagZ() const { return exagZ; }
double ParamPly::getDyn() const { return dyn; }

void ParamPly::setEchantInterval(int e) { echantInterval = e; }
void ParamPly::setNuages(const QVector<bool>& n) { nuages = n; }
QVector<bool>& ParamPly::modifNuages() { return nuages; }
void ParamPly::setDoFiltrage(bool d) { doFiltrage = d; }
void ParamPly::setDoFusion(bool doF) { doFusion = doF; }
void ParamPly::setDoEgalRadiom(bool doE) { doEgalRadiom = doE; }
void ParamPly::setMasques(const QList<pair<QString,QString> >& m) { masques = m; }
QList<pair<QString,QString> >& ParamPly::modifMasques() { return masques; }
void ParamPly::setDoBoite(bool doB) { doBoite = doB; }
void ParamPly::setBoite(const pair<QPoint,QPoint>& b) { boite = b; }
void ParamPly::setBinaire(bool b) { binaire = b; }
void ParamPly::setDoPly(bool doP) { doPly = doP; }
void ParamPly::setDoXyz(bool doX) { doXyz = doX; }
void ParamPly::setExagZ(double ex) { exagZ = ex; }
void ParamPly::setDyn(double dy) { dyn = dy; }

/////////////////////////////////////////////////////////////////////////////////////////////////


InterfOrtho::InterfOrtho(QWidget* parent, Assistant* help, ParamMain& param, QVector<CarteDeProfondeur>* cartesProfondeur):
	QDialog( parent ),
	paramMain( &param ),
	cartes( cartesProfondeur )
{
	setWindowModality(Qt::ApplicationModal);

	//liste de nuages
	QLabel* orthoLabel = new QLabel(conv(tr("Select orthoimage mosaic to compute :")));
	listWidget = new QListWidget;
	listWidget->setSelectionMode (QAbstractItemView::ExtendedSelection);
	listWidget->setPalette(QPalette(QColor(255,255,255)));
	for (int i=0; i<cartes->count(); i++) {
		if (cartes->at(i).getRepere() || !cartes->at(i).getDoOrtho()) continue;
		QListWidgetItem* li = new QListWidgetItem(cartes->at(i).getImageDeReference());
		listWidget->addItem(li);
	}
	listWidget->adjustSize();

	//autres options
		//filtrage automatique
	checkEgal = new QCheckBox(conv(tr("Radiometric equalization")));
	checkEgal->setToolTip(conv(tr("Make a radiometric equalization of images before computing the mosaic.")));
	checkEgal->setChecked(true);

	//boutons
	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	calButton = buttonBox->addButton (tr("Compute"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	assistant = help;
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(orthoLabel,0,Qt::AlignHCenter);
	mainLayout->addWidget(listWidget,0,Qt::AlignHCenter);
	mainLayout->addWidget(checkEgal,0,Qt::AlignHCenter);
	mainLayout->addSpacing(25);
	mainLayout->addWidget(buttonBox);
	mainLayout->addStretch();

	setLayout(mainLayout);
	setWindowTitle(conv(tr("Orthoimage mosaic computing")));
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
}

InterfOrtho::~InterfOrtho() {}

void InterfOrtho::calcClicked() {
	//orthos à calculer
	QList<QListWidgetItem *> l = listWidget->selectedItems();
	if (l.count()==0) {
		qMessageBox(this, tr("Error"), conv(tr("No depth maps selected.")));
		return;
	}
	int idx = 0;
	for (int i=0; i<cartes->count(); i++) {
		if (cartes->at(i).getRepere() || !cartes->at(i).getDoOrtho()) {
			(*cartes)[i].setOrthoCalculee(false);
			continue;
		}
		(*cartes)[i].setOrthoCalculee(listWidget->item(idx)->isSelected());
		idx++;		
	}
	hide();
	accept();
}

bool InterfOrtho::getEgaliser() const { return checkEgal->isChecked(); }

void InterfOrtho::helpClicked() { assistant->showDocumentation(assistant->pageInterfOrtho); }
