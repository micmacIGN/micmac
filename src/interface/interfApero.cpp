#include "interfApero.h"

using namespace std;


bool sortQlwip(QListWidgetItem* lwi1, QListWidgetItem* lwi2) { return (lwi1->text())<(lwi2->text()); }
QStringList listWidgetToStringList(const QList<QListWidgetItem*>& l) {
	QStringList l2;
	for (QList<QListWidgetItem*>::const_iterator it=l.begin(); it!=l.end(); it++)
		l2.push_back((*it)->text());
	return l2;
}

InterfApero::InterfApero(const ParamMain* pMain, QWidget* parent, Assistant* help) : 
	QDialog( parent ),
	assistant( help ),
	paramMain( pMain ),
	paramApero( paramMain->getParamApero() )
{
	done = false;
	//setWindowModality(Qt::ApplicationModal);
	if (paramApero.getLiberCalib().count()==0) {
		paramApero.modifLiberCalib().resize(paramMain->getParamPastis().getCalibFiles().count());
		for (int i=0; i<paramMain->getParamPastis().getCalibFiles().count(); i++)
			paramApero.modifLiberCalib()[i] = false;
	}

	tabWidget = new QTabWidget;
	tabWidget->setMovable (false);
	imgToOriTabA = new ImgToOriTabA (&paramApero, paramMain->getDossier(), &paramMain->getCorrespImgCalib());//, &icones
	tabWidget->addTab(imgToOriTabA, conv(tr("Images to be oriented")));
	maitresseTabA = new MaitresseTabA (&paramApero, paramMain, paramMain->getDossier());
	tabWidget->addTab(maitresseTabA, conv(tr("Master image")));
	referenceTabA = new ReferenceTabA (&paramApero, this, paramMain, assistant);
	tabWidget->addTab(referenceTabA, conv(tr("Orientation")));
	oriInitTabA = new OriInitTabA (&paramApero, paramMain->getDossier());
	tabWidget->addTab(oriInitTabA, conv(tr("Initial orientations")));
	autoCalibTabA = new AutoCalibTabA (&paramApero, paramMain->getDossier(), &paramMain->getCorrespImgCalib());//, &icones
	tabWidget->addTab(autoCalibTabA, conv(tr("Autocalibration")));
	multiEchelleTabA = 0;
	if (paramMain->getParamPastis().getCalibFiles().count()>1) {	//pas de multiéchelle s'il n'y a qu'une calibration
		multiEchelleTabA = new MultiEchelleTabA (&paramApero, paramMain->getParamPastis().getCalibFiles());
		tabWidget->addTab(multiEchelleTabA, conv(tr("Multiscale")));
	}
	liberCalibTabA = new LiberCalibTabA (&paramApero, paramMain->getParamPastis().getCalibFiles());
	tabWidget->addTab(liberCalibTabA, tr("Calibration"));
	ptsHomolTabA = new PtsHomolTabA (&paramApero);
	tabWidget->addTab(ptsHomolTabA, tr("Tie-points"));

	QDialogButtonBox* buttonBox = new QDialogButtonBox();
	precButton = buttonBox->addButton (conv(tr("Previous")), QDialogButtonBox::ActionRole);
	calButton = buttonBox->addButton (tr("Next"), QDialogButtonBox::AcceptRole);
	cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::RejectRole);
	helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(tabWidget);
	mainLayout->addWidget(buttonBox);

	setLayout(mainLayout);
	setWindowTitle(conv(tr("Orientation computing parameters")));
	adjustSize ();
	setMinimumSize(size());
	setMaximumSize(size());
	layout()->setSizeConstraint(QLayout::SetFixedSize);

	connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged()));
	connect(imgToOriTabA, SIGNAL(imgsSetChanged()), maitresseTabA, SLOT(imgsSetChanged()));
	connect(imgToOriTabA, SIGNAL(imgsSetChanged()), referenceTabA, SIGNAL(imgsSetChanged()));
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));
	connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));
	connect(precButton, SIGNAL(clicked()), this, SLOT(precClicked()));

	tabWidget->setCurrentIndex(0);
	tabChanged();
	done = true;
}
InterfApero::~InterfApero () {
	delete imgToOriTabA;
	delete maitresseTabA;
	delete referenceTabA;
	delete oriInitTabA;
	if (multiEchelleTabA!=0) delete multiEchelleTabA;
	delete liberCalibTabA;
	delete ptsHomolTabA;
}

void InterfApero::tabChanged() {
	updateInterfApero(tabWidget->currentIndex());
}

void InterfApero::updateInterfApero(int tab) {
	disconnect(calButton, SIGNAL(clicked()), this, SLOT(suivClicked()));
	disconnect (calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));

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
}

void InterfApero::precClicked() {
	tabWidget->setCurrentIndex(tabWidget->currentIndex()-1);
}
void InterfApero::suivClicked() {
	tabWidget->setCurrentIndex(tabWidget->currentIndex()+1);
}

void InterfApero::calcClicked() {
	//images à orienter
	if (paramApero.getImgToOri().count()==0) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("No selected images to be oriented.")));
		return;
	}

	//image maîtresse
	if (paramApero.getImgMaitresse().isEmpty()) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("A master image must be selected.")));
		return;
	}
	if (!paramApero.getImgToOri().contains(paramApero.getImgMaitresse())) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("The set of images to be oriented does not contain the selected master image.")));
		return;
	}

	//auto-calibration
	if (paramApero.getAutoCalib().count()==0 && autoCalibTabA->doAutoCalib()) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("No images selected for autocalibration while autocalibration computing is selected.")));
		return;
	}

	//multi-échelle
	if (paramApero.getMultiechelle() && paramApero.getCalibFigees().count()==0) {
		qMessageBox(this, conv(tr("Parameter error")), conv(tr("No calibrations selected for the first step of multiscale computing.")));
		return;
	}

	if (paramApero.getMultiechelle()) {
		//focale de l'image maîtresse
		int i = paramMain->findImg(paramApero.getImgMaitresse(),1);
		int j = 0;
		while (j<paramMain->getParamPastis().getCalibFiles().count()) {
			if (paramMain->getParamPastis().getCalibFiles().at(j).first==paramMain->getCorrespImgCalib().at(i).getCalibration())
				break;
			j++;
		}
		 if (!paramApero.getCalibFigees().contains(paramMain->getParamPastis().getCalibFiles().at(j).second)) {
			qMessageBox(this, conv(tr("Parameter error")), conv(tr("Although master image orientation should be estimated at the first step of multiscale computing, no calibrations were selected.")));
			return;
		}
	}

	//orientation absolue
	if (paramApero.getUserOrientation().getOrientMethode()==1 && paramApero.getUserOrientation().getBascOnPlan())
		/*if (!referenceTabA->masqueTrouve()) {
			qMessageBox(this, tr("Erreur de saisie"), conv(tr("Aucun masque du plan horizontal n'a été saisi.")));
			return;
		}*/
	//referenceTabA->saveMasques();
	if (!referenceTabA->renameDirBDDC()) return;
	if (paramApero.getUserOrientation().getOrientMethode()==1 && paramApero.getUserOrientation().getBascOnPlan())
		if (paramApero.getUserOrientation().getPoint1()==QPoint(-1,-1) || paramApero.getUserOrientation().getPoint2()==QPoint(-1,-1)) {
			qMessageBox(this, tr("Selection error"), conv(tr("No abscissa axes selected.")));
			return;
		}
	QString err = referenceTabA->saveImgAbsParam();
	if (!err.isEmpty()) {
		qMessageBox(this, tr("Selection error"), err);
		return;
	}
	if (paramApero.getUserOrientation().getOrientMethode()==2 && !paramApero.getImgToOri().contains(paramApero.getUserOrientation().getImageGeoref())) {
		qMessageBox(this, tr("Selection error"), conv(tr("The set of images to be oriented does not contain the image of known georeferencing.")));
		return;
	}
	if (paramApero.getUserOrientation().getOrientMethode()==3 && paramApero.getUserOrientation().getPointsGPS().isEmpty()) {
		qMessageBox(this, tr("Selection error"), conv(tr("No GCP file provided.")));
		return;
	}
	if (paramApero.getUserOrientation().getOrientMethode()==3 && paramApero.getUserOrientation().getAppuisImg().isEmpty()) {
		qMessageBox(this, tr("Selection error"), conv(tr("No GCP image measurement file provided.")));
		return;
	}

	//fin
	hide();
	accept();
}

void InterfApero::helpClicked() {
	if (tabWidget->currentWidget()==imgToOriTabA) assistant->showDocumentation(assistant->pageInterfApero);
	else if (tabWidget->currentWidget()==maitresseTabA) assistant->showDocumentation(assistant->pageInterfAperoMaitresse);
	else if (tabWidget->currentWidget()==referenceTabA) assistant->showDocumentation(assistant->pageInterfAperoReference);
	else if (tabWidget->currentWidget()==oriInitTabA) assistant->showDocumentation(assistant->pageInterfAperoOrInit);
	else if (tabWidget->currentWidget()==autoCalibTabA) assistant->showDocumentation(assistant->pageInterfAperoAutocalib);
	else if (tabWidget->currentWidget()==multiEchelleTabA) assistant->showDocumentation(assistant->pageInterfAperoMultiechelle);
	else if (tabWidget->currentWidget()==liberCalibTabA) assistant->showDocumentation(assistant->pageInterfAperoLibercalib);
	else if (tabWidget->currentWidget()==ptsHomolTabA) assistant->showDocumentation(assistant->pageInterfAperoPtshomol);
}

const ParamApero& InterfApero::getParamApero() const { return paramApero; }
bool InterfApero::isDone() { return done; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ImgToOriTabA::ImgToOriTabA(ParamApero* paramApero, QString dossier, const QVector<ParamImage>* correspImgCalib)//, const QVector<QIcon>* vignettes
{	
	dir = dossier;
	parametres = paramApero;
	//icones = vignettes;

	QLabel *label = new QLabel(conv(tr("Images to be oriented : ")));
	QLabel *label2 = new QLabel(tr("Remaining images : "));

	listWidget = new QListWidget;
	listWidget2 = new QListWidget;
	listWidget->setResizeMode(QListView::Adjust);
	listWidget2->setResizeMode(QListView::Adjust);
	for (int i=0; i<correspImgCalib->size(); i++) {
		QListWidgetItem* lwi = new QListWidgetItem(correspImgCalib->at(i).getImageTif());//icones->at(i), 
		if (parametres->getImgToOri().contains(correspImgCalib->at(i).getImageTif())) {
			listWidget->addItem(lwi);
		} else
			listWidget2->addItem(lwi);
	}
	listWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
	listWidget2->setSelectionMode(QAbstractItemView::ExtendedSelection);

	addButton = new QPushButton(QIcon(g_iconDirectory+"linguist-up.png"), QString());
	addButton->setToolTip(conv(tr("Add image on the list")));
	addButton->setEnabled(false);
	addButton->setMaximumSize (QSize(21,21));

	removeButton = new QPushButton(QIcon(g_iconDirectory+"linguist-down.png"), QString());
	removeButton->setToolTip(conv(tr("Remove image from the list")));
	removeButton->setEnabled(false);
	removeButton->setMaximumSize (QSize(21,21));

	connect(listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(imageSelected()));
	connect(listWidget2, SIGNAL(itemSelectionChanged()), this, SLOT(imageSelected2()));
	connect(addButton, SIGNAL(clicked()), this, SLOT(addClicked()));
	connect(removeButton, SIGNAL(clicked()), this, SLOT(removeClicked()));

	QGridLayout* widgetsLayout = new QGridLayout;
	widgetsLayout->addWidget(label,0,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(listWidget,1,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(addButton,2,0,1,1, Qt::AlignVCenter | Qt::AlignRight);
	widgetsLayout->addWidget(removeButton,2,1,1,1, Qt::AlignVCenter | Qt::AlignLeft);
	widgetsLayout->addWidget(label2,3,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(listWidget2,4,0,1,2, Qt::AlignCenter);
	widgetsLayout->setContentsMargins(15,30,15,30);	

	QGroupBox* mBox = new QGroupBox(this);
	mBox->setLayout(widgetsLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(mBox);//,0,Qt::AlignCenter
	//mainLayout->addStretch();
	mainLayout->setFormAlignment(Qt::AlignCenter);
	
	setLayout(mainLayout);
	adjustSize();
	setMinimumSize(size());
	setMaximumSize(size());
}
ImgToOriTabA::~ImgToOriTabA () {}

void ImgToOriTabA::imageSelected() {
	if (listWidget->selectedItems().count()>0) {
		removeButton->setEnabled(true);
		addButton->setEnabled(false);
		listWidget2->clearSelection();
	} else
		removeButton->setEnabled(false);
}

void ImgToOriTabA::imageSelected2() {
	if (listWidget2->selectedItems().count()>0) {
		addButton->setEnabled(true);
		removeButton->setEnabled(false);
		listWidget->clearSelection();
	} else
		addButton->setEnabled(false);
}

void ImgToOriTabA::addClicked() {
	//listWidget2 & parametres
	QList<QListWidgetItem *> l = listWidget2->selectedItems();	//images à ajouter
	QStringList l2 = listWidgetToStringList(l);
	for (QList<QListWidgetItem *>::const_iterator it=l.begin(); it!=l.end(); it++) {
		listWidget2->takeItem(listWidget2->row(*it));
	}	
	parametres->modifImgToOri() << l2;
	//listWidget
	if (listWidget->count()>0) l2 << listWidgetToStringList(listWidget->findItems(QString("tif"),Qt::MatchContains));
	qSort(l2.begin(),l2.end());
	listWidget->clear();	
	for (QStringList::const_iterator it=l2.begin(); it!=l2.end(); it++)
		listWidget->addItem((*it));
	//maj
	addButton->setEnabled(false);
	emit imgsSetChanged();	
}

void ImgToOriTabA::removeClicked() {
	//listWidget & parametres
	QList<QListWidgetItem *> l = listWidget->selectedItems();	//images à supprimer
	QStringList l2 = listWidgetToStringList(l);
	for (QList<QListWidgetItem*>::const_iterator it=l.begin(); it!=l.end(); it++) {
		listWidget->takeItem(listWidget->row(*it));
		parametres->modifImgToOri().removeAll((*it)->text());
	}
	//listWidget2
	if (listWidget2->count()>0) l2 << listWidgetToStringList(listWidget2->findItems(QString("tif"),Qt::MatchContains));
	qSort(l2.begin(),l2.end());
	listWidget2->clear();	
	for (QStringList::const_iterator it=l2.begin(); it!=l2.end(); it++)
		listWidget2->addItem((*it));	
	//maj
	removeButton->setEnabled(false);
	emit imgsSetChanged();	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MaitresseTabA::MaitresseTabA(ParamApero* paramApero, const ParamMain* pMain, QString dossier) : paramMain(pMain)
{	
	dir = dossier;
	parametres = paramApero;

	QLabel *label = new QLabel(conv(tr("Select master image : ")));

	listWidget = new QListWidget;
	listWidget->setResizeMode(QListView::Adjust);
	listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	for (int i=0; i<paramApero->getImgToOri().count(); i++) {
		listWidget->addItem(paramApero->getImgToOri().at(i));
		if (paramApero->getImgToOri().at(i)==paramApero->getImgMaitresse()) {
			listWidget->item(i)->setSelected(true);
		}
	}
	if (paramApero->getImgMaitresse().isEmpty() || listWidget->selectedItems().count()==0) {
		QString maitresse = calculeBestMaitresse();
		(*(listWidget->findItems(maitresse,Qt::MatchExactly).begin()))->setSelected(true);
	}

	apercuButton = new QToolButton;
	maitresseSelected();
	connect(listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(maitresseSelected()));

	QVBoxLayout* vboxLayout = new QVBoxLayout;
	vboxLayout->addWidget(label,0, Qt::AlignTop | Qt::AlignHCenter);
	vboxLayout->insertSpacing(2, 15);
	vboxLayout->addWidget(listWidget,0, Qt::AlignTop | Qt::AlignHCenter);
	vboxLayout->insertSpacing(4, 10);
	vboxLayout->addWidget(apercuButton,0, Qt::AlignTop | Qt::AlignHCenter);
	vboxLayout->addStretch();
	vboxLayout->setContentsMargins(15,30,15,30);	

	QGroupBox* mBox = new QGroupBox(this);
	mBox->setLayout(vboxLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(mBox);//,0,Qt::AlignCenter
	//mainLayout->addStretch();
	mainLayout->setFormAlignment(Qt::AlignCenter);
	
	setLayout(mainLayout);
	adjustSize();
	setMinimumSize(size());
	setMaximumSize(size());
}
MaitresseTabA::~MaitresseTabA () {}

QString MaitresseTabA::calculeBestMaitresse () {
	//recherche du nombre d'images connectées à chaque image et du nombre de points homologues	
	QVector<int> nbVois(listWidget->count(),0);
	QVector<int> nbHom(listWidget->count(),0);
	cTplValGesInit<string>  aTpl;
	char** argv = new char*[1];
	char c_str[] = "rthsrth";
	argv[0] = new char[strlen( c_str )+1];
	strcpy( argv[0], c_str );
	cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, paramMain->getDossier().toStdString(), aTpl );
	const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");
	for (int aK=0; aK<signed(aVN->size()) ; aK++) {
	  	pair<string,string> aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
		ElPackHomologue aPack = ElPackHomologue::FromFile( paramMain->getDossier().toStdString()+(*aVN)[aK]);
		QList<QListWidgetItem *> l1 = listWidget->findItems(aPair.first.c_str(),Qt::MatchExactly);
		if (l1.count()==0) continue;
		QList<QListWidgetItem *> l2 = listWidget->findItems(aPair.second.c_str(),Qt::MatchExactly);
		if (l2.count()==0) continue;
		int i1 = listWidget->row(*(l1.begin()));
		int i2 = listWidget->row(*(l2.begin()));
		nbVois[i1] ++;
		nbHom[i1] += aPack.size();
		nbVois[i2] ++;
		nbHom[i2] += aPack.size();
	}
	delete [] argv[0];
	delete [] argv;
	delete mICNM;
	//imges avec le plus d'images connectées
	QVector<int>::const_iterator maximum = max_element(nbVois.begin(), nbVois.end());
	int nb = count(nbVois.begin(), nbVois.end(), *maximum);
	if (nb==1) return listWidget->item(maximum-nbVois.begin())->text();
	//et avec le plus de points homologues
	QVector<int> bestImgs(listWidget->count(),0);
	QVector<int>::const_iterator it2=nbHom.begin();
	QVector<int>::iterator it3=bestImgs.begin();
	for (QVector<int>::const_iterator it=nbVois.begin(); it!=nbVois.end(); it++, it2++, it3++) {
		if (*it==*maximum) *it3 = *it2;
	}
	QVector<int>::const_iterator maximum2 = max_element(bestImgs.begin(), bestImgs.end());
	return listWidget->item(maximum2-bestImgs.begin())->text();
}

void MaitresseTabA::maitresseSelected() {
	parametres->setImgMaitresse( (*(listWidget->selectedItems().begin()))->text() );
	QString imageName = dir+paramMain->convertTifName2Couleur((*(listWidget->selectedItems().begin()))->text());
	QImage image( imageName );
	if (image.isNull()) {
		cout << tr("Fail to read image %1.").arg( imageName ).toStdString() << endl; 
		return;
	}
	image = image.scaled(150,150,Qt::KeepAspectRatio);
	apercuButton->setIconSize(image.size());
	apercuButton->setIcon(QPixmap::fromImage(image));
	apercuButton->adjustSize();
	adjustSize();
	//parentWidget()->adjustSize();
}

void MaitresseTabA::imgsSetChanged() {
	disconnect(listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(maitresseSelected()));
	listWidget->clear();
	for (QStringList::const_iterator it=parametres->getImgToOri().begin(); it!=parametres->getImgToOri().end(); it++)
		listWidget->addItem((*it));
	QList<QListWidgetItem *> l2 = listWidget->findItems(parametres->getImgMaitresse(),Qt::MatchExactly);
	connect(listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(maitresseSelected()));
	if (l2.count()!=0) (*(l2.begin()))->setSelected(true);
	else (*(listWidget->findItems(calculeBestMaitresse(),Qt::MatchExactly).begin()))->setSelected(true);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ReferenceTabA::ReferenceTabA (ParamApero* paramApero, InterfApero* parentWindow, const ParamMain* pMain, Assistant* help) :
	QScrollArea(),
	dir( pMain->getDossier() ),
	parametres( paramApero ),
	paramMain( pMain ),
	assistant( help ),
	parent( parentWindow ),
	paintInterfAppui( 0 ),
	fileDialogSommets( 0 )
{
	//sélection du type d'orientation
	radioAucun = new QRadioButton(conv(tr("Relative orientation (no GCP)")));
	radioPlan = new QRadioButton(tr("User orientation"));
	radioImageAbs = new QRadioButton(conv(tr("Absolute orientation of an image (not recommended)")));
	radioAppuis = new QRadioButton(conv(tr("Georeferencing with GCP")));
	radioSommets = new QRadioButton(conv(tr("GPS coordinates of pose summit")));
	
	QButtonGroup* buttonGroup = new QButtonGroup;
	buttonGroup->addButton(radioAucun,0);
	buttonGroup->addButton(radioPlan,1);
	buttonGroup->addButton(radioImageAbs,2);
	buttonGroup->addButton(radioAppuis,3);
	buttonGroup->addButton(radioSommets,4);
	buttonGroup->setExclusive(true);
	connect(buttonGroup, SIGNAL(buttonClicked(int)),this, SLOT(radioClicked(int)));

	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	//cas de l'orientation manuelle
		//options
	checkDoPlanDir = new QCheckBox(conv(tr("Define a rotation")));
	checkDoPlanDir->setChecked(parametres->getUserOrientation().getBascOnPlan());
	checkDoEchelle = new QCheckBox(conv(tr("Define a scale")));	
	checkDoEchelle->setChecked(parametres->getUserOrientation().getFixEchelle());
	connect(checkDoPlanDir, SIGNAL(stateChanged(int)), this, SLOT(doPlanDirChecked()));
	connect(checkDoEchelle, SIGNAL(stateChanged(int)), this, SLOT(doEchelleChecked()));

		//saisie du plan et de la direction
	QLabel* planLabel = new QLabel(tr("Horizontal plan"));
	QFont font;
	font.setBold(true);
	planLabel->setFont(font);
	masqueWidget = new MasqueWidget(paramMain, assistant, false, false, 0, parametres->getUserOrientation().getImgMasque(), QString("_MasqPlan"));
		mapper = new QSignalMapper(); 	
		connect(mapper, SIGNAL(mapped(int)),this, SLOT(updateParam(int)));
	connect(masqueWidget, SIGNAL(updateParam()), mapper, SLOT(map()));
	connect(this, SIGNAL(imgsSetChanged()), masqueWidget, SLOT(imgsSetChanged()));
	mapper->setMapping(masqueWidget, 0);

	QLabel* directionLabel = new QLabel(tr("Direction"));
	directionLabel->setFont(font);
	directionWidget = new DirectionWidget(paramMain, parametres->getImgToOri(), assistant,
						pair<QString,QString>(parametres->getUserOrientation().getImage1(),parametres->getUserOrientation().getImage2()), 2,
						pair<QPoint,QPoint>(parametres->getUserOrientation().getPoint1(),parametres->getUserOrientation().getPoint2()),
						parametres->getUserOrientation().getAxe());
	connect(directionWidget, SIGNAL(updateParam()), mapper, SLOT(map()));
	mapper->setMapping(directionWidget, 1);
		
	QVBoxLayout *planDirLayout = new QVBoxLayout;
	planDirLayout->addWidget(planLabel);
	planDirLayout->addWidget(masqueWidget->getMasqueBox());
	planDirLayout->addWidget(directionLabel);
	planDirLayout->addWidget(directionWidget->getBox());
	planDirLayout->addStretch(1);

	planDirBox = new QGroupBox(this);
	planDirBox->setLayout(planDirLayout);

		//échelle
	echelleWidget = new EchelleWidget(paramMain, 4, parametres->getImgToOri(), parametres->getImgToOri(), assistant,
						pair<QVector<QString>,QVector<QPoint> >(parametres->getUserOrientation().getImages(),parametres->getUserOrientation().getPoints()),
						parametres->getUserOrientation().getDistance());
	connect(echelleWidget, SIGNAL(updateParam()), mapper, SLOT(map()));
	mapper->setMapping(echelleWidget, 2);

		//mise en page
	QVBoxLayout *manuLayout = new QVBoxLayout;
	manuLayout->addWidget(checkDoPlanDir);
	manuLayout->addWidget(planDirBox);
	manuLayout->addWidget(checkDoEchelle);
	manuLayout->addWidget(echelleWidget->getBox());
	manuLayout->addStretch(1);

	manuBox = new QGroupBox(this);
	manuBox->setLayout(manuLayout);

	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	//cas de l'orientation absolue d'une image
		//choisir l'image orientée
	QLabel* imgAbsLabel = new QLabel(conv(tr("Image of known georeferencing : ")));
	imgAbsCombo = new QComboBox;
	imgAbsCombo->setMinimumWidth(150);
	imgAbsCombo->addItems(paramMain->getParamApero().getImgToOri());
	if (parametres->getUserOrientation().getOrientMethode()==2)
		imgAbsCombo->setCurrentIndex(imgAbsCombo->findText(parametres->getUserOrientation().getImageGeoref()));

	QHBoxLayout *imgAbsRefLayout = new QHBoxLayout;
	imgAbsRefLayout->addWidget(imgAbsLabel);
	imgAbsRefLayout->addWidget(imgAbsCombo);
	imgAbsRefLayout->addStretch();

	QGroupBox* imgAbsRefBox = new QGroupBox;
	imgAbsRefBox->setFlat(true);
	imgAbsRefBox->setAlignment(Qt::AlignLeft);
	imgAbsRefBox->setLayout(imgAbsRefLayout);

		//type de données	
	QLabel* radioAbsLabel = new QLabel(conv(tr("Georeferencing parameters :")));
	radioFichier = new QRadioButton(conv(tr("Load a file")));
	radioHand = new QRadioButton(tr("enter them manually"));
	if (parametres->getUserOrientation().getOrientMethode()==2) {
		if (!parametres->getUserOrientation().getGeorefFile().isEmpty()) radioFichier->setChecked(true);
		else radioHand->setChecked(false);
	} else 
		radioFichier->setChecked(false);	

	QGridLayout *radioAbsLayout = new QGridLayout;
	radioAbsLayout->addWidget(radioAbsLabel,0,0,1,2,Qt::AlignHCenter);
	radioAbsLayout->addWidget(radioFichier,1,0,1,1,Qt::AlignHCenter);	
	radioAbsLayout->addWidget(radioHand,1,1,1,1,Qt::AlignHCenter);	

	QGroupBox* radioAbsBox = new QGroupBox;
	radioAbsBox->setLayout(radioAbsLayout);
	connect(radioFichier, SIGNAL(clicked(bool)), this, SLOT(radioAbsChecked(bool)));
	connect(radioHand, SIGNAL(clicked(bool)), this, SLOT(radioAbsChecked(bool)));

		//import d'un fichier
	QLabel* fichierAbsLabel = new QLabel(tr("Georeferencing file : "));
	fichierAbsEdit = new QLineEdit;
	fichierAbsEdit->setMinimumWidth(150);
	fichierAbsEdit->setEnabled(false);
	if (parametres->getUserOrientation().getOrientMethode()==2 && !parametres->getUserOrientation().getGeorefFile().isEmpty())
		fichierAbsEdit->setText(parametres->getUserOrientation().getGeorefFile());
	QPushButton* fichierAbsButton = new QPushButton(tr("..."));
	fichierAbsButton->setToolTip(tr("Open a georeferencing file"));
	fichierAbsButton->setMaximumSize (QSize(21,16));

	QHBoxLayout *fichierAbsLayout = new QHBoxLayout;
	fichierAbsLayout->addWidget(fichierAbsLabel);
	fichierAbsLayout->addWidget(fichierAbsEdit);
	fichierAbsLayout->addWidget(fichierAbsButton);
	fichierAbsLayout->addStretch();

	fichierAbsBox = new QGroupBox;
	fichierAbsBox->setFlat(true);
	fichierAbsBox->setAlignment(Qt::AlignLeft);
	fichierAbsBox->setLayout(fichierAbsLayout);
	connect(fichierAbsButton, SIGNAL(clicked(bool)), this, SLOT(fichierAbsClicked()));
	
		//saisie manuelle de l'orientation -> formulaire
	QLabel* centerAbsLabel = new QLabel(tr("summit"));
	centerAbsEdit.resize(3);
	for (int i=0; i<3; i++) {
		centerAbsEdit[i] = new QLineEdit;
		centerAbsEdit[i]->setMinimumWidth(150);
		if (parametres->getUserOrientation().getOrientMethode()==2 && parametres->getUserOrientation().getGeorefFile().isEmpty())
			centerAbsEdit[i]->setText(QVariant(parametres->getUserOrientation().getCentreAbs().at(i)).toString());
	}

	QLabel* rotationAbsLabel = new QLabel(tr("rotation"));
	rotationAbsEdit.resize(9);
	for (int i=0; i<9; i++) {
		rotationAbsEdit[i] = new QLineEdit;
		rotationAbsEdit[i]->setMinimumWidth(150);
		if (parametres->getUserOrientation().getOrientMethode()==2 && parametres->getUserOrientation().getGeorefFile().isEmpty())
			centerAbsEdit[i]->setText(QVariant(parametres->getUserOrientation().getRotationAbs().at(i)).toString());
	}	

	QGridLayout *formAbsLayout = new QGridLayout;
	formAbsLayout->addWidget(centerAbsLabel,0,0,1,1,Qt::AlignHCenter);
	for (int i=0; i<3; i++) formAbsLayout->addWidget(centerAbsEdit[i],0,i+1,1,1,Qt::AlignHCenter);
	formAbsLayout->addWidget(rotationAbsLabel,1,0,1,1,Qt::AlignHCenter);	
	for (int i=0; i<9; i++)  formAbsLayout->addWidget(rotationAbsEdit[i],1+i%3,1+i/3,1,1,Qt::AlignHCenter);

	formAbsBox = new QGroupBox;
	formAbsBox->setLayout(formAbsLayout);

		//échelle
	echelleWidget2 = new EchelleWidget(paramMain, 4, parametres->getImgToOri(), parametres->getImgToOri(), assistant, pair<QVector<QString>,QVector<QPoint> >(parametres->getUserOrientation().getImages(),parametres->getUserOrientation().getPoints()), parametres->getUserOrientation().getDistance());
	connect(echelleWidget2, SIGNAL(updateParam()), mapper, SLOT(map()));
	mapper->setMapping(echelleWidget2, 3);

		//mise en page
	QVBoxLayout *imgAbsLayout = new QVBoxLayout;
	imgAbsLayout->addWidget(imgAbsRefBox);
	imgAbsLayout->addWidget(radioAbsBox);
	imgAbsLayout->addWidget(fichierAbsBox);
	imgAbsLayout->addWidget(formAbsBox);
	imgAbsLayout->addWidget(echelleWidget2->getBox());
	fichierAbsBox->hide();
	formAbsBox->hide();
	if (parametres->getUserOrientation().getOrientMethode()==2) {
		if (parametres->getUserOrientation().getGeorefFile().isEmpty()) formAbsBox->show();
		else fichierAbsBox->show();
	}
	imgAbsLayout->addStretch(1);

	imgAbsBox = new QGroupBox(this);
	imgAbsBox->setLayout(imgAbsLayout);

	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	//cas du géoréférencement par points d'appui
		//import d'un fichier de points terrain
	QLabel* fileAppLabel = new QLabel(conv(tr("GCP file : ")));
	fileAppEdit = new QLineEdit;
	fileAppEdit->setMinimumWidth(150);
	fileAppEdit->setEnabled(false);
	fileAppButton = new QPushButton(tr("..."));
	fileAppButton->setToolTip(tr("Open a file"));
	fileAppButton->setMaximumSize (QSize(21,16));

	QHBoxLayout *fileAppLayout = new QHBoxLayout;
	fileAppLayout->addWidget(fileAppLabel);
	fileAppLayout->addWidget(fileAppEdit);
	fileAppLayout->addWidget(fileAppButton);
	fileAppLayout->addStretch();

	QGroupBox* fileAppBox = new QGroupBox;
	fileAppBox->setFlat(true);
	fileAppBox->setAlignment(Qt::AlignLeft);
	fileAppBox->setLayout(fileAppLayout);

		//import d'un fichier de mesures
	QLabel* fileMesLabel = new QLabel(conv(tr("Image measure file : ")));
	fileMesEdit = new QLineEdit;
	fileMesEdit->setMinimumWidth(150);
	fileMesEdit->setEnabled(false);
	fileMesButton = new QPushButton(tr("..."));
	fileMesButton->setToolTip(tr("Open a file"));
	fileMesButton->setMaximumSize (QSize(21,16));

		//saisie des points
	QLabel* fileSaisieLabel = new QLabel(conv(tr("or measure them : ")));
	fileSaisieButton = new QPushButton(QIcon(g_iconDirectory+"designer-edit-resources-button.png"), QString());
	fileSaisieButton->setToolTip(conv(tr("Measure points on graphic window")));
	fileSaisieButton->setMaximumSize (QSize(21,16));

	QHBoxLayout *fileMesLayout = new QHBoxLayout;
	fileMesLayout->addWidget(fileMesLabel);
	fileMesLayout->addWidget(fileMesEdit);
	fileMesLayout->addWidget(fileMesButton);
	fileMesLayout->addWidget(fileSaisieLabel);
	fileMesLayout->addWidget(fileSaisieButton);
	fileMesLayout->addStretch();

	QGroupBox* fileMesBox = new QGroupBox;
	fileMesBox->setFlat(true);
	fileMesBox->setAlignment(Qt::AlignLeft);
	fileMesBox->setLayout(fileMesLayout);

		//mise en boîte
	QVBoxLayout *appuiLayout = new QVBoxLayout;
	appuiLayout->addWidget(fileAppBox);
	appuiLayout->addWidget(fileMesBox);
	if (parametres->getUserOrientation().getOrientMethode()==3) {
		if (parametres->getUserOrientation().getOrientMethode()==3 && !parametres->getUserOrientation().getPointsGPS().isEmpty()) fileAppEdit->setText(parametres->getUserOrientation().getPointsGPS());
		if (parametres->getUserOrientation().getOrientMethode()==3 && !parametres->getUserOrientation().getAppuisImg().isEmpty()) fileMesEdit->setText(parametres->getUserOrientation().getAppuisImg());
	}
	appuiLayout->addStretch(1);

	appuiBox = new QGroupBox(this);
	appuiBox->setLayout(appuiLayout);

	connect(fileAppButton, SIGNAL(clicked(bool)), this, SLOT(appuisClicked()));
	connect(fileMesButton, SIGNAL(clicked(bool)), this, SLOT(mesAppClicked()));
	connect(fileSaisieButton, SIGNAL(clicked(bool)), this, SLOT(saisieAppClicked()));

	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	//cas du géoréférencement par sommets GPS
	QLabel* fileSommetsLabel = new QLabel(conv(tr("GPS coordinates of pose summit file : ")));
	fileSommetsEdit = new QLineEdit;
	fileSommetsEdit->setMinimumWidth(150);
	fileSommetsEdit->setEnabled(false);
	fileSommetsButton = new QPushButton(tr("..."));
	fileSommetsButton->setToolTip(tr("Open a file or a directory"));
	fileSommetsButton->setMaximumSize (QSize(21,16));
	if (parametres->getUserOrientation().getOrientMethode()==4 && QDir(dir+QString("Ori-BDDC")).exists()) {
		fileSommetsEdit->setText(dir+QString("Ori-BDDC"));
		parametres->modifUserOrientation().setPointsGPS(dir+QString("Ori-BDDC"));
	}


		//mise en boîte
	QHBoxLayout *fileSommetsLayout = new QHBoxLayout;
	fileSommetsLayout->addWidget(fileSommetsLabel);
	fileSommetsLayout->addWidget(fileSommetsEdit);
	fileSommetsLayout->addWidget(fileSommetsButton);
	fileSommetsLayout->addStretch();

	sommetsBox = new QGroupBox(this);
	sommetsBox->setLayout(fileSommetsLayout);

	connect(fileSommetsButton, SIGNAL(clicked(bool)), this, SLOT(sommetsClicked()));

	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	//mise en page
	QVBoxLayout *radioLayout = new QVBoxLayout;
	radioLayout->addWidget(radioAucun,0,Qt::AlignLeft);
	radioLayout->addWidget(radioPlan,0,Qt::AlignLeft);
	radioLayout->addWidget(manuBox,0,Qt::AlignLeft);
	radioLayout->addWidget(radioImageAbs,0,Qt::AlignLeft);
	radioLayout->addWidget(imgAbsBox,0,Qt::AlignLeft);
	radioLayout->addWidget(radioAppuis,0,Qt::AlignLeft);
	radioLayout->addWidget(appuiBox,0,Qt::AlignLeft);
	radioLayout->addWidget(radioSommets,0,Qt::AlignLeft);
	radioLayout->addWidget(sommetsBox,0,Qt::AlignLeft);
	radioLayout->addStretch(1);
	manuBox->hide();
	imgAbsBox->hide();
	appuiBox->hide();
	sommetsBox->hide();

	QGroupBox* radioBox = new QGroupBox(this);
	radioBox->setObjectName(conv(tr("Select an orientation method : ")));
	radioBox->setLayout(radioLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(radioBox);
	mainLayout->setFormAlignment(Qt::AlignCenter);

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	//setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	radioClicked(parametres->getUserOrientation().getOrientMethode());
	buttonGroup->button(parametres->getUserOrientation().getOrientMethode())->setChecked(true);	//dans cet ordre
	doPlanDirChecked();
	doEchelleChecked();
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
}
ReferenceTabA::~ReferenceTabA () {
	if (fileDialogSommets!=0) delete fileDialogSommets;
	if (paintInterfAppui!=0) delete paintInterfAppui;
}

void ReferenceTabA::resizeEvent(QResizeEvent*) {
	QSize size1 = QApplication::desktop()->availableGeometry().size()+parent->size()-size();
	QSize size2 = resizableWidget->size();
	QSize size3( min(size1.width(),size2.width()) , min(size1.height()*3/4,size2.height()) );
	setMinimumSize(size3);
}

void ReferenceTabA::radioClicked(int idx) {
	switch (idx) {	//ne pas regrouper sinon pb de hide() et de redimensionnement
		case 0 : parametres->modifUserOrientation().setOrientMethode(0);
				manuBox->hide();
				imgAbsBox->hide();
				appuiBox->hide();
				sommetsBox->hide();
				break;
		case 1 : parametres->modifUserOrientation().setOrientMethode(1);
				imgAbsBox->hide();
				appuiBox->hide();
				sommetsBox->hide();
				manuBox->show();
				break;
		case 2 : parametres->modifUserOrientation().setOrientMethode(2);
				manuBox->hide();
				appuiBox->hide();
				sommetsBox->hide();
				imgAbsBox->show();
				break;
		case 3 : parametres->modifUserOrientation().setOrientMethode(3);
				manuBox->hide();
				imgAbsBox->hide();
				sommetsBox->hide();
				appuiBox->show();
				break;
		case 4 : parametres->modifUserOrientation().setOrientMethode(4);
				manuBox->hide();
				imgAbsBox->hide();
				appuiBox->hide();
				sommetsBox->show();
				break;
	}
}

void ReferenceTabA::updateParam(int idx) {
	switch (idx) {
		case 0 : masqueWidget->updateParam(parametres);
			break;
		case 1 : directionWidget->updateParam(parametres);
			break;
		case 2 : echelleWidget->updateParam(parametres);
			break;
		case 3 : echelleWidget2->updateParam(parametres);
			break;
	}
}

//orientation manuelle xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

void ReferenceTabA::doPlanDirChecked() {
	if (checkDoPlanDir->isChecked()) planDirBox->show();
	else planDirBox->hide();
	parametres->modifUserOrientation().setBascOnPlan(checkDoPlanDir->isChecked());
}
void ReferenceTabA::doEchelleChecked() {
	if (checkDoEchelle->isChecked()) echelleWidget->getBox()->show();
	else echelleWidget->getBox()->hide();
	parametres->modifUserOrientation().setFixEchelle(checkDoEchelle->isChecked());
}

//orientation absolue d'une image xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

void ReferenceTabA::radioAbsChecked(bool) {
	if (radioFichier->isChecked()) {
		formAbsBox->hide();
		fichierAbsBox->show();
	} else if (radioHand->isChecked()) {
		fichierAbsBox->hide();
		formAbsBox->show();
	}
}

void ReferenceTabA::fichierAbsClicked() {
	FileDialog fileDialog(this, tr("Open a georeferencing file"), dir, tr("Georeferencing file (*.xml);;") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) {
		fileNames = fileDialog.selectedFiles();
	} else return;
  	if (fileNames.size()==0)
		return;
  	if (*(fileNames.begin())==fichierAbsEdit->text())
		return;
	QString fichier = *(fileNames.begin());
	//chemin absolu
	fichier = QDir(dir).absoluteFilePath(fichier);
	//on vérifie que le nom du fichier est lisible (à cause des accents)
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}
	fichierAbsEdit->setText(fichier);
}

QString ReferenceTabA::saveImgAbsParam() {
	if (!radioImageAbs->isChecked()) return QString();
	parametres->modifUserOrientation().setOrientMethode(2);
	parametres->modifUserOrientation().setImageGeoref(imgAbsCombo->currentText());
	if (radioFichier->isChecked())
		parametres->modifUserOrientation().setGeorefFile(fichierAbsEdit->text());
	else {
		parametres->modifUserOrientation().setGeorefFile(QString());
		for (int i=0; i<3; i++) {
			if (centerAbsEdit[i]->text().isEmpty()) return conv(tr("A coordinate of the absolute orientation summit is empty."));
			bool ok = false;
			parametres->modifUserOrientation().modifCentreAbs()[i] = centerAbsEdit[i]->text().toDouble(&ok);
			if (!ok) return conv(tr("A coordinate of the absolute orientation summit is unvalid."));
		}
		for (int i=0; i<9; i++) {
			if (rotationAbsEdit[i]->text().isEmpty()) return conv(tr("An element of the absolute orientation rotation is empty."));
			bool ok = false;
			parametres->modifUserOrientation().modifRotationAbs()[i] = rotationAbsEdit[i]->text().toDouble(&ok);
			if (!ok) return conv(tr("An element of the absolute orientation rotation is unvalid."));
		}
	}
	return QString();
}

//géoréférencement par points d'appui xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

void ReferenceTabA::appuisClicked() {
	FileDialog fileDialog(this, tr("Open a GCP file"), dir, tr("All files (*);;") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) {
		fileNames = fileDialog.selectedFiles();
	} else return;
  	if (fileNames.size()==0)
		return;
  	if (*(fileNames.begin())==fileAppEdit->text())
		return;
	QString fichier = *(fileNames.begin());
	//chemin absolu
	fichier = QDir(dir).absoluteFilePath(fichier);
	//on vérifie que le nom du fichier est lisible (à cause des accents)
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}
	fileAppEdit->setText(fichier);
	parametres->modifUserOrientation().setPointsGPS(fichier);
}

void ReferenceTabA::mesAppClicked() {
	FileDialog fileDialog(this, tr("Open an image measure file"), dir, tr("All files (*);;") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) {
		fileNames = fileDialog.selectedFiles();
	} else return;
  	if (fileNames.size()==0)
		return;
  	if (*(fileNames.begin())==fileMesEdit->text())
		return;
	QString fichier = *(fileNames.begin());
	//chemin absolu
	fichier = QDir(dir).absoluteFilePath(fichier);
	//on vérifie que le nom du fichier est lisible (à cause des accents)
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"), conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}
	fileMesEdit->setText(fichier);
	parametres->modifUserOrientation().setAppuisImg(fichier);
}

void ReferenceTabA::saisieAppClicked() {
	//lecture
	QList<QString> points;
	QString err = FichierAppuiGPS::lire(fileAppEdit->text(), points);
	if (!err.isEmpty()) {
		qMessageBox(this, tr("Read error"), err);
		return;
	}
	QVector<QVector<QPoint> > pointsAppui(paramMain->getCorrespImgCalib().count(),QVector<QPoint>(points.count(),QPoint(-1,-1)));
	if (!fileMesEdit->text().isEmpty()) {
		err = FichierAppuiImage::lire(fileMesEdit->text(), paramMain, points, pointsAppui);
		if (!err.isEmpty()) {
			qMessageBox(this, tr("Read error"), err);
			return;
		}
	}

	//affichage et saisie
	if (paintInterfAppui!=0) delete paintInterfAppui;
	paintInterfAppui = new PaintInterfAppui(paramMain, assistant, points, pointsAppui, this);
	if (!paintInterfAppui->getDone()) return;
	int rep = paintInterfAppui->exec();
	if (rep != QDialog::Accepted) return;
	pointsAppui = paintInterfAppui->getPointsAppui();
	//enregistrement
	QString fileName = FileDialog::getSaveFileName(this, tr("Save GCP measures"), dir, conv(tr("XML files (*.xml)")));
	if (fileName.isEmpty()) return;
	if (fileName.contains("/")) fileName = fileName.section("/",-1,-1);
	if (fileName.contains(".")) fileName = fileName.section (".",0,-2);
	if (fileName.isEmpty()) fileName = QString("mesureAppui");
	fileName = paramMain->getDossier() + fileName + QString(".xml");
	fileMesEdit->setText(fileName);
	if (!FichierAppuiImage::ecrire(fileName, paramMain, points, pointsAppui)) {
		qMessageBox(this, tr("Read error"), conv(tr("Fail to save GCP measure file.")));
		return;
	}
	parametres->modifUserOrientation().setAppuisImg(fileName);
	delete paintInterfAppui;
	paintInterfAppui = 0;
}

//géoréférencement par coordonnées GPS des sommets xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//

void ReferenceTabA::sommetsClicked() {
	if (fileDialogSommets!=0) delete fileDialogSommets;
	fileDialogSommets = new FileDialog(this, conv(tr("Open a GPS coordinate file")), dir, tr("All files (*);; all directories (*);;") );
	
	connect(fileDialogSommets, SIGNAL(filterSelected(QString)), this, SLOT(filterSelected(QString)));
	fileDialogSommets->setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialogSommets->exec())
		fileNames = fileDialogSommets->selectedFiles();
	else return;
  	if (fileNames.size()==0)
		return;
  	if (*(fileNames.begin())==fileAppEdit->text())
		return;
	QString fichier = *(fileNames.begin());
	if (!QDir(fichier).exists() && !QFile(fichier).exists()) {
		qMessageBox(this, tr("Read error"),conv(tr("File or directory %1 does not exist.")).arg(fichier));	
		return;
	}
	//chemin absolu
	fichier = QDir(dir).absoluteFilePath(fichier);
	//on vérifie que le nom du fichier est lisible (à cause des accents)
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}
	if (QDir(fichier).exists() && fichier.right(1)!=QString("/")) fichier = fichier + QString("/");
	fileSommetsEdit->setText(fichier);
	parametres->modifUserOrientation().setPointsGPS(fichier);
}
void ReferenceTabA::filterSelected(const QString& filtre) {
//permet de choisir à la fois des fichiers texte (liste des points GPS) et des dossiers (coordonnées GPS formatées en xml par pose) avec la même QFileDialog
	if (filtre==tr("All directories (*)"))
		fileDialogSommets->setFileMode(QFileDialog::Directory);
	else
		fileDialogSommets->setFileMode(QFileDialog::ExistingFile);
}

bool ReferenceTabA::renameDirBDDC() {
//renomme le dossier des sommets GPS en Ori-BDDC
	QString dirBDDC = parametres->getUserOrientation().getPointsGPS();
	if (parametres->getUserOrientation().getOrientMethode()!=4) return true;
	if (!QDir(dirBDDC).exists()) return true;	//fichier à convertir avec aperoThread
	if (dirBDDC==dir+QString("Ori-BDDC/")) return true;	//rien à faire
	if (QDir(dir+QString("Ori-BDDC")).exists()) {
		int i = 0;
		while (QDir(dir+QString("Ori-BDDC%1").arg(i)).exists()) i++;
		QDir(dir).rename(QString("Ori-BDDC"),QString("Ori-BDDC%1").arg(i));
	}
	parametres->modifUserOrientation().setPointsGPS(dir+QString("Ori-BDDC/"));
	return QFile(dirBDDC).rename(dir+QString("Ori-BDDC"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


OriInitTabA::OriInitTabA(ParamApero* paramApero, QString dossier) : QWidget(), dir(dossier), parametres(paramApero)
{	
	checkBox = new QCheckBox(conv(tr("Use an initial orientation")));
	checkBox->setToolTip(conv(tr("If checked, use a former result as initial orientation.")));
	checkBox->setChecked(parametres->getUseOriInit());

	QLabel *label = new QLabel(conv(tr("Initial orientation files : ")));
		QFont font;
		font.setBold(true);
	label->setFont(font);

	textEdit = new QLineEdit;
	if (QDir(dir+QString("Ori-Initiale")).exists()) {
		textEdit->setText(QString("Ori-Initiale"));
		paramApero->setDirOriInit(QString("Ori-Initiale"));
	} else if (!paramApero->getDirOriInit().isEmpty() && QDir(dir+paramApero->getDirOriInit()).exists())
		textEdit->setText(paramApero->getDirOriInit());
	else {
		textEdit->setText(QString("Ori-F"));
		paramApero->setDirOriInit(QString("Ori-F"));
	}
	textEdit->setEnabled(false);

	dirButton = new QPushButton(QString("..."));
	dirButton->setToolTip(conv(tr("Select files")));
	dirButton->setEnabled(true);
	dirButton->setMaximumSize (QSize(21,21));

	QHBoxLayout* oriLayout = new QHBoxLayout;
	oriLayout->addWidget(label,0);
	oriLayout->addWidget(textEdit,0);
	oriLayout->addWidget(dirButton,0);

	oriBox = new QGroupBox;
	oriBox->setLayout(oriLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(checkBox);
	mainLayout->addWidget(oriBox);
	setLayout(mainLayout);

	connect(checkBox, SIGNAL(stateChanged(int)), this, SLOT(boxChecked()));
	connect(dirButton, SIGNAL(clicked()), this, SLOT(dirClicked()));

	boxChecked();
	adjustSize();
	mainLayout->setSizeConstraint(QLayout::SetFixedSize);
}
OriInitTabA::~OriInitTabA () {}

void OriInitTabA::boxChecked() {
	if (checkBox->checkState()==Qt::Checked) oriBox->show();
	else oriBox->hide();
	parametres->setUseOriInit(checkBox->checkState()==Qt::Checked);
}

void OriInitTabA::dirClicked() {
	FileDialog* fileDialog = new FileDialog(this, conv(tr("Select initial orientations")), dir, tr("All directories (*);;") );
	fileDialog->setAcceptMode(QFileDialog::AcceptOpen);
	fileDialog->setFileMode(QFileDialog::Directory);

	QStringList dirNames;
	if (fileDialog->exec()) dirNames = fileDialog->selectedFiles();
	else return;
  	if (dirNames.size()==0) return;

	QString dirName = *(dirNames.begin());
	dirName = QDir(dirName).absolutePath();
  	if (dirName==dir+textEdit->text()) return;
	if (!checkPath(dirName)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(dirName));	
		return;
	}
	if (dirName.right(1)==QString("/")) dirName.resize(dirName.count()-1);
	textEdit->setText(dirName.section("/",-1,-1));
	if (dirName.section("/",0,-2)!=dir) QFile(dirName).copy(dir+textEdit->text());
	textEdit->setText(dirName.section("/",-1,-1));
	parametres->setDirOriInit(textEdit->text());
	delete fileDialog;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


AutoCalibTabA::AutoCalibTabA(ParamApero* paramApero, QString dossier, const QVector<ParamImage>* correspImgCalib) : QWidget()//, const QVector<QIcon>* vignettes
{	
	dir = dossier;
	parametres = paramApero;
	images = correspImgCalib;
//	icones = vignettes;

	//sélection du type d'orientation
	radioOui = new QRadioButton(conv(tr("Compute an autocalibration before computing orientations")));
	radioOui->setToolTip(conv(tr("Compute an autocalibration on an image set before computing orientations to provide more accurate calibrations.\nNB : Calibrations are estimated while computing orientations too.")));
	radioNon = new QRadioButton(conv(tr("No autocalibration before computing orientations")));	

	QLabel *label = new QLabel(conv(tr("Autocalibration images : ")));
	QLabel *label2 = new QLabel(tr("Remaining images : "));
	QFont font;
	font.setBold(true);
	label->setFont(font);
	label2->setFont(font);

	listWidget = new QListWidget;
	listWidget2 = new QListWidget;
	listWidget->setResizeMode(QListView::Adjust);
	listWidget2->setResizeMode(QListView::Adjust);
	for (int i=0; i<correspImgCalib->size(); i++) {
		QListWidgetItem* lwi = new QListWidgetItem(correspImgCalib->at(i).getImageTif());//icones->at(i), 
		if (parametres->getAutoCalib().contains(correspImgCalib->at(i).getImageTif())) {
			listWidget->addItem(lwi);
		} else
			listWidget2->addItem(lwi);
	}
	listWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
	listWidget2->setSelectionMode(QAbstractItemView::ExtendedSelection);

	addButton = new QPushButton(QIcon(g_iconDirectory+"linguist-up.png"), QString());
	addButton->setToolTip(conv(tr("Add an image on the list")));
	addButton->setEnabled(false);
	addButton->setMaximumSize (QSize(21,21));

	removeButton = new QPushButton(QIcon(g_iconDirectory+"linguist-down.png"), QString());
	removeButton->setToolTip(conv(tr("Remove an image from the list")));
	removeButton->setEnabled(false);
	removeButton->setMaximumSize (QSize(21,21));

	QGridLayout* widgetsLayout = new QGridLayout;
	widgetsLayout->addWidget(label,0,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(listWidget,1,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(addButton,2,0,1,1, Qt::AlignVCenter | Qt::AlignRight);
	widgetsLayout->addWidget(removeButton,2,1,1,1, Qt::AlignVCenter | Qt::AlignLeft);
	widgetsLayout->addWidget(label2,3,0,1,2, Qt::AlignCenter);
	widgetsLayout->addWidget(listWidget2,4,0,1,2, Qt::AlignCenter);
	widgetsLayout->setContentsMargins(15,30,15,30);	

	autoBox = new QGroupBox(this);
	autoBox->setLayout(widgetsLayout);

	connect(radioOui, SIGNAL(clicked()), this, SLOT(radioClicked()));
	connect(radioNon, SIGNAL(clicked()), this, SLOT(radioClicked()));
	connect(listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(imageSelected()));
	connect(listWidget2, SIGNAL(itemSelectionChanged()), this, SLOT(imageSelected2()));
	connect(addButton, SIGNAL(clicked()), this, SLOT(addClicked()));
	connect(removeButton, SIGNAL(clicked()), this, SLOT(removeClicked()));

	//mise en page
	QVBoxLayout *radioLayout = new QVBoxLayout;
	radioLayout->addWidget(radioOui,0,Qt::AlignLeft);
	radioLayout->addWidget(autoBox,0,Qt::AlignLeft);
	radioLayout->addWidget(radioNon,0,Qt::AlignLeft);
	radioLayout->addStretch(1);
	//autoBox->hide();

	QGroupBox* radioBox = new QGroupBox(this);
	radioBox->setLayout(radioLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(radioBox);
	mainLayout->setFormAlignment(Qt::AlignCenter);
	setLayout(mainLayout);

	radioOui->setChecked(parametres->getAutoCalib().count()>0);
	radioNon->setChecked(parametres->getAutoCalib().count()==0);
	radioClicked();
	adjustSize();
	mainLayout->setSizeConstraint(QLayout::SetFixedSize);
}
AutoCalibTabA::~AutoCalibTabA () {}

void AutoCalibTabA::radioClicked() {
	if (radioOui->isChecked())
		autoBox->show();
	else {
		autoBox->hide();
		if (parametres->modifAutoCalib().count()>0) {
			parametres->modifAutoCalib().clear();
			listWidget->clear();
			listWidget2->clear();
			for (int i=0; i<images->count(); i++) {
				QListWidgetItem* lwi = new QListWidgetItem(images->at(i).getImageTif());//icones->at(i), 
				listWidget2->addItem(lwi);
		
			}
		}
	}
}

void AutoCalibTabA::imageSelected() {
	if (listWidget->selectedItems().count()>0) {
		removeButton->setEnabled(true);
		addButton->setEnabled(false);
		listWidget2->clearSelection();
	} else
		removeButton->setEnabled(false);
}

void AutoCalibTabA::imageSelected2() {
	if (listWidget2->selectedItems().count()>0) {
		addButton->setEnabled(true);
		removeButton->setEnabled(false);
		listWidget->clearSelection();
	} else
		addButton->setEnabled(false);
}

void AutoCalibTabA::addClicked() {
	QList<QListWidgetItem *> l = listWidget2->selectedItems();
	for (int i=0; i<l.count(); i++) {
		QListWidgetItem* lwi = new QListWidgetItem(*(l[i]));
		listWidget->addItem(lwi);
		listWidget2->takeItem(listWidget2->row(l.at(i)));	
		parametres->modifAutoCalib().push_back(lwi->text());
	}
	addButton->setEnabled(false);	
}

void AutoCalibTabA::removeClicked() {
	QList<QListWidgetItem *> l = listWidget->selectedItems();
	for (int i=0; i<l.count(); i++) {
		QListWidgetItem* lwi = new QListWidgetItem(*(l[i]));
		listWidget2->addItem(lwi);
		listWidget->takeItem(listWidget->row(l.at(i)));
		parametres->modifAutoCalib().removeAll(lwi->text());
	}
	removeButton->setEnabled(false);
}

bool AutoCalibTabA::doAutoCalib() const { return radioOui->isChecked(); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MultiEchelleTabA::MultiEchelleTabA (ParamApero* paramApero, const QList<std::pair<QString, int> >& calibFiles) : calibrations(&calibFiles) {
	parametres = paramApero;

	//Multi-échelle ?
	checkMulti = new QCheckBox(conv(tr("Two step orientation")));
	checkMulti->setToolTip(conv(tr("If selected\n1 - estimates short focal length image orientation\n2 - fixes these orientations and estimates long focal length image orientation")));
	if (parametres->getCalibFigees().count()>0)
		checkMulti->setChecked (true);
	else
		checkMulti->setChecked (false);
	parametres->setMultiechelle( !(checkMulti->checkState()==Qt::Unchecked) );

	//sélection des courtes focales
	QLabel *label1 = new QLabel(conv(tr("Select calibrations for estimation first step (short focal lengths) :")));

	listWidget1 = new QListWidget;
	listWidget1->setResizeMode(QListView::Adjust);
	for (int i=0; i<calibrations->count(); i++) {
		listWidget1->addItem(QVariant(calibrations->at(i).second).toString());
		listWidget1->item(listWidget1->count()-1)->setTextAlignment(Qt::AlignHCenter);
		if (parametres->getCalibFigees().contains(calibrations->at(i).second))
			listWidget1->item(listWidget1->count()-1)->setSelected(true);
		else
			listWidget1->item(listWidget1->count()-1)->setSelected(false);
	}
	listWidget1->setSelectionMode(QAbstractItemView::ExtendedSelection);
	listWidget1->adjustSize();
	listWidget1->setMaximumWidth(50);
	/*if (precCalibFigees!=0)		
		*(parametres->calibFigees) = *precCalibFigees;*/

	//sélection des focales longues
	QLabel *label2 = new QLabel(conv(tr("Remained calibration for estimation second step (long focal lengths) :")));

	listWidget2 = new QListWidget;
	listWidget2->setResizeMode(QListView::Adjust);
/*	for (int i=0; i<calibrations->count(); i++) {
		if (precCalibFigees!=0 && precgetCalibFigees().contains(calibrations->at(i).second)) continue;
		listWidget2->addItem(QVariant(calibrations->at(i).second).toString());
		listWidget2->item(listWidget2->count()-1)->setTextAlignment(Qt::AlignHCenter);
	}*/
	dispList2();
	listWidget2->setSelectionMode(QAbstractItemView::NoSelection);
//	listWidget2->adjustSize();
	//listWidget2->setMaximumSize(50,listWidget2->count()*20);
	listWidget2->setMaximumWidth(50);

	QPalette pal = palette();
	QColor color = pal.color(QPalette::Window);
	pal.setColor(QPalette::Base, color);
	listWidget2->setPalette(pal);

	//mise en page
	QVBoxLayout *multiLayout = new QVBoxLayout;
	multiLayout->addWidget(label1);
	multiLayout->addWidget(listWidget1,0,Qt::AlignHCenter);
	multiLayout->addWidget(label2);
	multiLayout->addWidget(listWidget2,0,Qt::AlignHCenter);
	multiLayout->addStretch(1);

	multiBox = new QGroupBox(this);
	multiBox->setLayout(multiLayout);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(checkMulti, 0, Qt::AlignCenter);
	mainLayout->addWidget(multiBox, 0, Qt::AlignCenter);
	mainLayout->addStretch();
	
	setLayout(mainLayout);
	adjustSize();

	connect(checkMulti, SIGNAL(stateChanged(int)), this, SLOT(displayMulti()));
	connect(listWidget1, SIGNAL(itemSelectionChanged()), this, SLOT(dispList2()));
	if (parametres->getCalibFigees().count()==0) multiBox->hide();
}
MultiEchelleTabA::~MultiEchelleTabA () {}

void MultiEchelleTabA::displayMulti() {
	if (checkMulti->checkState()==Qt::Unchecked) {
		multiBox->hide();
	} else {
		multiBox->show();
	}
	parametres->setMultiechelle( !(checkMulti->checkState()==Qt::Unchecked) );
}

void MultiEchelleTabA::dispList2() {
	listWidget2->clear();
	parametres->modifCalibFigees().clear();
	QList<QListWidgetItem *> l = listWidget1->selectedItems();
	for (int i=0; i<calibrations->count(); i++) {
		int b = false;
		for (int j=0; j<l.count(); j++) {
			if (QVariant(calibrations->at(i).second).toString()==l.at(j)->text()) {
				b = true;
				parametres->modifCalibFigees().push_back(calibrations->at(i).second);
				break;
			}
			if (!b) {
				listWidget2->addItem(QVariant(calibrations->at(i).second).toString());
				listWidget2->item(listWidget2->count()-1)->setTextAlignment(Qt::AlignHCenter);
			}
		}
	}
	listWidget2->adjustSize();
	//listWidget2->setMaximumSize(50,listWidget2->count()*20);
	listWidget2->setMaximumWidth(50);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


LiberCalibTabA::LiberCalibTabA (ParamApero* paramApero, const QList<std::pair<QString, int> >& calibFiles) : calibrations(&calibFiles) {
	parametres = paramApero;

	checkLiber = new QCheckBox*[calibrations->count()];
	for (int i=0; i<calibrations->count(); i++) {
		checkLiber[i] = new QCheckBox (QVariant(calibrations->at(i).second).toString()+QString(" mm"));
		if (parametres->modifLiberCalib()[i]) checkLiber[i]->setChecked (true);
		else checkLiber[i]->setChecked (false);
		connect(checkLiber[i], SIGNAL(stateChanged(int)), this, SLOT(liberClicked()));
	}	

	QVBoxLayout *liberLayout = new QVBoxLayout;
	for (int i=0; i<calibrations->count(); i++)
		liberLayout->addWidget(checkLiber[i],0,Qt::AlignCenter);
	liberLayout->addStretch(1);

	QGroupBox* liberBox = new QGroupBox(tr("Dissociate calibrations with these focal lengths :"), this);
	liberBox->setToolTip(conv(tr("If selected, estimates a single calibration for each image of the set (case of a focus alteration)")));
	liberBox->setLayout(liberLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(liberBox);
	mainLayout->setFormAlignment(Qt::AlignCenter);
	
	setLayout(mainLayout);
	adjustSize();
	liberClicked();

}
LiberCalibTabA::~LiberCalibTabA () {
	delete [] checkLiber;
}

void LiberCalibTabA::liberClicked() {
	for (int i=0; i<calibrations->count(); i++)
		parametres->modifLiberCalib()[i] = (checkLiber[i]->checkState()==2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


PtsHomolTabA::PtsHomolTabA (ParamApero* paramApero) {
	parametres = paramApero;

	checkFiltr = new QCheckBox(tr("Filter tie-points"));
	checkFiltr->setToolTip(conv(tr("If selected, filters tie-points to reduce their count and speed up computing")));
	checkFiltr->setChecked(parametres->getFiltrage());
	connect(checkFiltr, SIGNAL(stateChanged(int)), this, SLOT(filtrClicked()));

	checkCalc3D = new QCheckBox(tr("Computing tie-points in 3D"));
	checkCalc3D->setToolTip(conv(tr("If selected, computes tie-points in 3D and displays them in the 3D view (can take a quite long time)")));
	checkCalc3D->setChecked(parametres->getCalcPts3D());
	connect(checkCalc3D, SIGNAL(stateChanged(int)), this, SLOT(calcPt3DClicked()));

	checkExport3D = new QCheckBox(tr("Export 3D tie-points to ply format"));
	checkExport3D->setToolTip(conv(tr("If selected, exports 3D tie-point and camera positions into a ply file")));
	checkExport3D->setChecked(parametres->getExportPts3D());
	connect(checkExport3D, SIGNAL(stateChanged(int)), this, SLOT(exportPt3DClicked()));

	QVBoxLayout *ptsHomolLayout = new QVBoxLayout;
	ptsHomolLayout->addWidget(checkFiltr,0,Qt::AlignCenter);
	ptsHomolLayout->addWidget(checkCalc3D,0,Qt::AlignCenter);
	ptsHomolLayout->addWidget(checkExport3D,0,Qt::AlignCenter);
	ptsHomolLayout->addStretch(1);
	if (!checkCalc3D->isChecked()) checkExport3D->hide();

	QGroupBox* ptsHomolBox = new QGroupBox;
	ptsHomolBox->setLayout(ptsHomolLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(ptsHomolBox);
	mainLayout->setFormAlignment(Qt::AlignCenter);
	
	setLayout(mainLayout);
	adjustSize();
	filtrClicked();
}
PtsHomolTabA::~PtsHomolTabA () {}

void PtsHomolTabA::filtrClicked() { parametres->setFiltrage( (checkFiltr->checkState()==2) ); }
void PtsHomolTabA::calcPt3DClicked() { parametres->setCalcPts3D( (checkCalc3D->checkState()==2) ); }
void PtsHomolTabA::exportPt3DClicked() { parametres->setExportPts3D( (checkExport3D->checkState()==2) ); }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MasqueWidget::MasqueWidget(const ParamMain* param, Assistant* help, bool mm, bool mmasq, QPushButton* vue3DButton, const QString& image, const QString& postfx):
	QWidget(),
	masque( 0 ),
	assistant( help ),
	paramMain( param ),
	dir( param->getDossier() ),
	paintInterf( 0 ),
	imageFond( image ),
	masqueFile( QString() ),
	postfixe( postfx ),
	micmac( mm ),
	micmacMasque( mmasq )
{
	if (!imageFond.isEmpty())
		if (!calcMasqueFile())
			qMessageBox(this, tr("Read error"), conv(tr("Fail to find corresponding mask name.")));

	//choisir une image
	if (!micmacMasque) {
		QLabel* imageLabel = new QLabel(conv(tr("Reference image : ")));
		imageCombo = new QComboBox;
		imageCombo->setMinimumWidth(150);
		for (int i=0; i<paramMain->getParamApero().getImgToOri().count(); i++)
			imageCombo->addItem(paramMain->getParamApero().getImgToOri().at(i));
		if (!imageFond.isEmpty()) imageCombo->setCurrentIndex(imageCombo->findText(imageFond.section("/",-1,-1)));
		else imageCombo->setCurrentIndex(-1);
		connect(imageCombo, SIGNAL(currentIndexChanged(QString)), this, SLOT(comboChanged(QString)));

		QHBoxLayout *imageLayout = new QHBoxLayout;
		imageLayout->addWidget(imageLabel);
		imageLayout->addWidget(imageCombo);
		if (vue3DButton!=0) imageLayout->addWidget(vue3DButton);
		imageLayout->addStretch();

		imageBox = new QGroupBox;
		imageBox->setFlat(true);
		imageBox->setAlignment(Qt::AlignLeft);
		imageBox->setLayout(imageLayout);
	}

	//créer ou ou ouvrir un masque
	radioNvu = new QRadioButton(conv(tr("Draw a new mask")));
	radioOpen = new QRadioButton(tr("Open an existing mask"));
	radioNvu->setChecked(false);	
	if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) radioOpen->setChecked(true);
	else radioOpen->setChecked(false);
	connect(radioNvu, SIGNAL(clicked(bool)), this, SLOT(choiceClicked()));
	connect(radioOpen, SIGNAL(clicked(bool)), this, SLOT(choiceClicked()));

	QVBoxLayout *radioLayout = new QVBoxLayout;	
	radioLayout->addWidget(radioNvu,0,Qt::AlignHCenter);
	radioLayout->addWidget(radioOpen,0,Qt::AlignHCenter);
	radioLayout->addStretch(1);

	radioBox = new QGroupBox;
	radioBox->setLayout(radioLayout);

	//ouvrir un masque
	QLabel* openLabel = new QLabel(tr("Mask :"));
	openEdit = new QLineEdit;
	openEdit->setMinimumWidth(150);
	openEdit->setEnabled(false);
	if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) openEdit->setText(masqueFile);
	openButton = new QPushButton(tr("..."));
	openButton->setToolTip(tr("Open a mask"));
	openButton->setMaximumSize (QSize(21,16));
	connect(openButton, SIGNAL(clicked()), this, SLOT(openClicked()));

	QHBoxLayout *openLayout = new QHBoxLayout;
	openLayout->addWidget(openLabel);
	openLayout->addWidget(openEdit);
	openLayout->addWidget(openButton);
	openLayout->addStretch();

	openBox = new QGroupBox;
	openBox->setFlat(true);
	openBox->setAlignment(Qt::AlignLeft);
	openBox->setLayout(openLayout);

	//modifier le masque
	modifButton = new QPushButton(tr("Modify mask"));
	modifButton->setMaximumSize (QSize(133,24));
	connect(modifButton, SIGNAL(clicked()), this, SLOT(modifClicked()));

	//enregistrer le masque
	QLabel* saveLabel = new QLabel(tr("Mask saving : "));
	saveEdit = new QLineEdit;
	saveEdit->setMinimumWidth(150);
	saveEdit->setEnabled(false);
	if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) saveEdit->setText(masqueFile);

	QHBoxLayout *saveLayout = new QHBoxLayout;
	saveLayout->addWidget(saveLabel);
	saveLayout->addWidget(saveEdit);
	saveLayout->addStretch();

	saveBox = new QGroupBox;
	saveBox->setFlat(true);
	saveBox->setAlignment(Qt::AlignLeft);
	saveBox->setLayout(saveLayout);

	//layouts
	QVBoxLayout* masqueLayout = new QVBoxLayout;
	if (!micmacMasque) masqueLayout->addWidget(imageBox);
	masqueLayout->addWidget(radioBox);
	masqueLayout->addWidget(openBox);
	masqueLayout->addWidget(modifButton,0,Qt::AlignHCenter);
	masqueLayout->addWidget(saveBox);
	masqueLayout->addStretch();

	masqueBox = new QGroupBox;
	masqueBox->setFlat(false);
	masqueBox->setAlignment(Qt::AlignCenter);
	masqueBox->setContentsMargins(0,10,0,10);
	masqueBox->setLayout(masqueLayout);

	updateInterface(Begin);
	adjustSize();
}
MasqueWidget::~MasqueWidget() {
	if (paintInterf!=0) delete paintInterf;
}

void MasqueWidget::imgsSetChanged() {
	imageCombo->clear();
	for (int i=0; i<paramMain->getParamApero().getImgToOri().count(); i++)
		imageCombo->addItem(paramMain->getParamApero().getImgToOri().at(i));
	int idx = imageCombo->findText(imageFond);
	if (idx==-1) {
		imageFond = QString();
		emit updateParam();
		updateInterface(Begin);
	} else
		imageCombo->setCurrentIndex(idx);
}

bool MasqueWidget::calcMasqueFile() {
	if (!micmacMasque) masqueFile = imageFond.section(".",0,-2)+postfixe+QString(".tif");
	else {
		bool ok;
		masqueFile = paramMain->getDossier()+QString("Masque_%1.tif").arg(paramMain->getNumImage(imageFond,&ok,true));
		if (!ok) masqueFile = paramMain->getDossier()+QString("Masque_%1.tif").arg(paramMain->getNumImage(imageFond,&ok,false));
		if (!ok) return false;
	}
	return true;
}

void MasqueWidget::comboChanged(QString txt) {
	setImageFond(txt);
}
void MasqueWidget::setImageFond(const QString& img) {
	imageFond = dir+img;
	if (!calcMasqueFile()) return;
	if (QFile(masqueFile).exists()) {
		openEdit->setText(masqueFile);
		saveEdit->setText(masqueFile);
	}
	emit updateParam();
	updateInterface(Image);
}

void MasqueWidget::choiceClicked() {
	if (imageFond.isEmpty()) {
		qMessageBox(this, tr("Parameter error"),conv(tr("An image must be selected first.")));	
		return;
	}
	if (radioNvu->isChecked()) {
		updateInterface(NewMasque);
		showPainter();
		updateInterface(Enreg);
	} else if (radioOpen->isChecked())
		updateInterface(OpenMasque);
}

void MasqueWidget::openClicked() {
	FileDialog fileDialog(this, tr("Open a mask"), dir, tr("Mask (*.tif);;Mask (*.tiff)") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) {
		fileNames = fileDialog.selectedFiles();
	} else return;
  	if (fileNames.size()==0)
		return;
  	if (*(fileNames.begin())==openEdit->text())
		return;
	if (paintInterf!=0) {
		delete paintInterf;
		paintInterf = 0;
	}
	masque = 0;
	QString fichier = *(fileNames.begin());

	//on vérifie que le nom du fichier est lisible (à cause des accents)
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}

	//si c'est le fichier non tuilé, on le remplace par le fichier tuilé
	if (fichier.right(12)==QString("nontuile.tif") && QFile(fichier.left(fichier.size()-12)+QString(".tif")).exists()) {
		fichier = fichier.left(fichier.size()-12)+QString(".tif");
	}	

	openEdit->setText(fichier);
	if (fichier!=masqueFile) QFile(fichier).copy(masqueFile);
	saveEdit->setText(masqueFile);
	saveClicked();	//pour le fichier de géoréférencement
}

void MasqueWidget::showPainter (QString masquePrec) {
	//lancement de l'outil de saisie du masque
	if (paintInterf!=0) {
		delete paintInterf;
		paintInterf = 0;
	}
	masque = 0;

	//conversion au format tif non tuilé
	if (!QFile(paramMain->convertTifName2Couleur(imageFond)).exists()) {
		QString err = convert2Rgba(imageFond, false, paramMain->convertTifName2Couleur(imageFond));
		if (!err.isEmpty()) {
			qMessageBox(this, conv(tr("Execution error")), err);
			QApplication::restoreOverrideCursor();
			return;
		}
	}
	if (masquePrec!=QString() && !QFile(imgNontuilee(masquePrec)).exists()) {
		QString err = convert2Rgba(masqueFile, true, imgNontuilee(masquePrec));
		if (!err.isEmpty()) {
			qMessageBox(this, conv(tr("Execution error")), err);
			QApplication::restoreOverrideCursor();
			return;
		}
	}

	if (micmac) paintInterf = new PaintInterfCorrel(paramMain->convertTifName2Couleur(imageFond), paramMain, assistant, this, masquePrec);
	else paintInterf = new PaintInterfPlan(paramMain->convertTifName2Couleur(imageFond), paramMain, assistant, this, true, masquePrec);
	if (!paintInterf->getDone()) return;
	paintInterf->show();
	if (paintInterf->exec() != QDialog::Accepted) return;
	masque = paintInterf->getMaskImg();
	saveClicked();
}

void MasqueWidget::modifClicked() {
	if (radioNvu->isChecked()) {
		paintInterf->show();
	} else {
		if (paintInterf==0) {	//import d'un masque existant
			showPainter(masqueFile);
		} else {
			paintInterf->show();
		}
	}
	if (paintInterf->exec() != QDialog::Accepted) return;
	masque = paintInterf->getMaskImg();
	saveClicked();
}

QString MasqueWidget::convert2Rgba(const QString& tuiledFile, bool toMask, const QString& newFile) {
//convertit le fichier pour être lisible par Qt et si toMask, convertit le fichier au format du masque de drawMask (masque en vert transparent)
	QString saveFile = (newFile.isEmpty())? (imgNontuilee(tuiledFile)) : newFile;
	//if (QFile(saveFile).exists()) return QString();
	deleteFile(saveFile);
	QString commande = noBlank(applicationPath()) + QString("/lib/tiff2rgba ") + noBlank(tuiledFile) + QString(" ") + noBlank(saveFile);
	if (execute(commande)!=0)
		return conv(tr("Fail to convert image %1 to untiled tif format.")).arg(tuiledFile);
	if (!toMask) return QString();
	QImage img(saveFile);
	QImage img2(img.size(), img.format());
	for (int x=0; x<img2.width(); x++) {
	for (int y=0; y<img2.height(); y++) {
		if (img.pixel(x,y)==QColor(0,0,0).rgb()) img2.setPixel(x,y,QColor(0,0,0,0).rgb());	//partie noire -> transparente
		else img2.setPixel(x,y,QColor(0,255,0,125).rgb());	//partie blanche -> vert semi-transparent
	}}
	if (!img2.save(saveFile)) return conv(tr("Fail to save modified mask %1.")).arg(saveFile);
	return QString();
}

void MasqueWidget::saveClicked() {
	QString refMasque = masqueFile.section(".",0,-2)+QString(".xml");

	//masque
	QSize size;
	if (masque!=0) {	//new ou modif
		if (QFile(masqueFile).exists()) QFile(masqueFile).remove();
		if (!QFile(applicationPath()+QString("/masquetempo.tif")).rename(masqueFile)) {
			if (!checkPath(masqueFile)) {
				qMessageBox(this, tr("Read error"),conv(tr("Fail to read write %1.\nCheck there are no accents in path.")).arg(dir+masqueFile));	
				return;
			} else {
				qMessageBox(this, tr("Read error"),conv(tr("Fail to write file %1.")).arg(dir+masqueFile));	
				return;
			}
		}
		QString masqueFile2 = imgNontuilee(masqueFile);
		if (QFile(masqueFile2).exists()) QFile(masqueFile2).remove();
		if (QFile(imgNontuilee(applicationPath()+QString("/masquetempo.tif"))).exists()) QFile(imgNontuilee(applicationPath()+QString("/masquetempo.tif"))).remove();	//pas de masque non tuilé si création
		size = QSize(masque->sz().x,masque->sz().y);
	} else {
			ELISE_fp fp;
			if (!fp.ropen(masqueFile.toStdString().c_str(),true)) {
				qMessageBox(this, tr("Read error"), conv(tr("Fail to read mask %1.")).arg(masqueFile));
				return;
			}
			fp.close();
		//juste open -> on récupère la taille du masque
		char *buf = new char[masqueFile.count()];
		sprintf(buf,"%s",masqueFile.toStdString().c_str());
		Tiff_Im m(buf);
		size = QSize(m.sz().x,m.sz().y);
		delete [] buf;
	}

	//fichier de référencement du masque
	bool ok;
	paramMain->getNumImage(imageFond,&ok,false);
	if (ok) {	//repère image
		if (!QFile(refMasque).exists()) {
			ParamMasqueXml paramMasqueXml(masqueFile, QString(), size);
			if(!(FichierMasque::ecrire(refMasque, paramMasqueXml))) {
				qMessageBox(this, conv(tr("Read error")), conv(tr("Fail to create mask referencing file.")));
				return;
			}
		}
	}

	saveEdit->setText(masqueFile);
	if (masque!=0) qMessageBox(this, tr("Saving"), conv(tr("Mask saved.")));
	emit updateParam();
	updateInterface(Enreg);
}

void MasqueWidget::updateInterface(Mode mode) {
	currentMode = mode;
	switch (mode) {
		case Begin :
			if (paintInterf!=0) {
				delete paintInterf;
				paintInterf = 0;
			}
			masque = 0;

			if (!micmacMasque) imageBox->setVisible(true);
			radioBox->setVisible(false);
			radioNvu->setChecked(false);
			radioOpen->setChecked(false);
			if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) updateInterface(OpenMasque);
			else {
				if (micmacMasque || !imageFond.isEmpty()) updateInterface(Image);
				else {
					openBox->setVisible(false);
					openEdit->clear();
					modifButton->setVisible(false);
					saveBox->setVisible(false);
					saveEdit->clear();
				}
			}
			break;
		case Image :
			if (paintInterf!=0) {
				delete paintInterf;
				paintInterf = 0;
			}
			masque = 0;

			if (!micmacMasque) imageBox->setVisible(true);
			radioBox->setVisible(true);
			radioNvu->setChecked(false);
			if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) updateInterface(OpenMasque);
			else {
				radioOpen->setChecked(false);
				openBox->setVisible(false);
				openEdit->clear();
				modifButton->setVisible(false);
				saveBox->setVisible(false);
				saveEdit->clear();
			}
			break;
		case NewMasque :
			if (paintInterf!=0) {
				delete paintInterf;
				paintInterf = 0;
			}
			masque = 0;

			if (!micmacMasque) imageBox->setVisible(true);
			radioBox->setVisible(true);
			radioNvu->setChecked(true);
			radioOpen->setChecked(false);
			openBox->setVisible(false);
			openEdit->clear();
			modifButton->setVisible(false);
			saveBox->setVisible(false);
			saveEdit->clear();
			break;
		case OpenMasque :
			if (paintInterf!=0) {
				delete paintInterf;
				paintInterf = 0;
			}
			masque = 0;

			if (!micmacMasque) imageBox->setVisible(true);
			radioBox->setVisible(true);
			radioNvu->setChecked(false);
			radioOpen->setChecked(true);
			openBox->setVisible(true);
			if (!masqueFile.isEmpty() && QFile(masqueFile).exists()) updateInterface(Enreg);
			else {
				openEdit->clear();
				modifButton->setVisible(false);
				saveBox->setVisible(false);
				saveEdit->clear();
			}
			break;

		case Enreg :
			modifButton->setVisible(true);
			saveBox->setVisible(true);
			emit updateParam();
			break;
	}
}

const MasqueWidget::Mode& MasqueWidget::getCurrentMode() const { return currentMode; }
QGroupBox* MasqueWidget::getMasqueBox() { return masqueBox; }


void MasqueWidget::updateParam(ParamApero* parametres) {
	parametres->modifUserOrientation().setImgMasque(imageFond);
}
void MasqueWidget::updateParam(CarteDeProfondeur* parametres, bool repere) {
	if (repere) parametres->setImgRepMasq(imageFond);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


DirectionWidget::DirectionWidget(const ParamMain* pMain, const QStringList & liste, Assistant* help, const std::pair<QString,QString>& imgPrec, int N, const std::pair<QPoint,QPoint>& ptsPrec, const QPoint& axePrec) : QWidget(), paintInterfSegment(0), dir(pMain->getDossier()), paramMain(pMain), assistant(help), nbList(N)
{
	QFont font;
	font.setBold(true);
	QLabel* directionLabel = new QLabel(tr("Direction"));
	directionLabel->setFont(font);

			//image
	QLabel* imageDirLabel = new QLabel(tr("Images onto draw axis :"));
	imageDirCombo1 = new QComboBox;
	imageDirCombo1->addItems(liste);
	if (!imgPrec.first.isEmpty()) imageDirCombo1->setCurrentIndex(imageDirCombo1->findText(imgPrec.first));
	else imageDirCombo1->setCurrentIndex(0);
	if (nbList==2) {
		imageDirCombo2 = new QComboBox;
		imageDirCombo2->addItems(liste);
		if (!imgPrec.second.isEmpty()) imageDirCombo2->setCurrentIndex(imageDirCombo2->findText(imgPrec.second));
		else imageDirCombo2->setCurrentIndex(0);
	}

	imageDirButton = new QPushButton(QIcon(g_iconDirectory+"linguist-check-on.png"), QString());
	imageDirButton->setToolTip(tr("Open these images"));
	imageDirButton->setMaximumSize (QSize(21,21));
	connect(imageDirButton, SIGNAL(clicked()), this, SLOT(imageDirClicked()));

	QHBoxLayout *imageDirLayout = new QHBoxLayout;
	imageDirLayout->addWidget(imageDirLabel);
	imageDirLayout->addWidget(imageDirCombo1);
	if (nbList==2) imageDirLayout->addWidget(imageDirCombo2);
	imageDirLayout->addWidget(imageDirButton);
	imageDirLayout->addStretch();

	QGroupBox* imageDirBox = new QGroupBox;
	imageDirBox->setFlat(true);
	imageDirBox->setAlignment(Qt::AlignLeft);
	imageDirBox->setLayout(imageDirLayout);

			//point
	QLabel* segmentLabel = new QLabel(tr("Direction :"));
	pointsEdit.resize(4);
	for (int i=0; i<4; i++) {
		pointsEdit[i] = new QLineEdit;
		pointsEdit[i]->setMaximumWidth(100);
		pointsEdit[i]->setEnabled(false);
	}	
	QLabel* point1Label = new QLabel(tr("point 1"));
	QLabel* point2Label = new QLabel(tr("point 2"));
		if (ptsPrec.first!=QPoint(-1,-1)) {
			pointsEdit[0]->setText(QVariant(ptsPrec.first.x()).toString());
			pointsEdit[1]->setText(QVariant(ptsPrec.first.y()).toString());
		}
		if (ptsPrec.second!=QPoint(-1,-1)) {
			pointsEdit[2]->setText(QVariant(ptsPrec.second.x()).toString());
			pointsEdit[3]->setText(QVariant(ptsPrec.second.y()).toString());
		}

	QHBoxLayout *segmentDirLayout = new QHBoxLayout;
	segmentDirLayout->addWidget(segmentLabel);
	segmentDirLayout->addWidget(point1Label);
	segmentDirLayout->addWidget(pointsEdit[0]);
	segmentDirLayout->addWidget(pointsEdit[1]);
	segmentDirLayout->addWidget(point2Label);
	segmentDirLayout->addWidget(pointsEdit[2]);
	segmentDirLayout->addWidget(pointsEdit[3]);

	QGroupBox* segmentDirBox = new QGroupBox;
	segmentDirBox->setFlat(true);
	segmentDirBox->setAlignment(Qt::AlignLeft);
	segmentDirBox->setLayout(segmentDirLayout);

		//axe
	QLabel* axeLabel = new QLabel(tr("Axis direction :"));
	radioX = new QRadioButton(QString("x"));
	radioY = new QRadioButton(QString("y"));
	radioMX = new QRadioButton(QString("-x"));
	radioMY = new QRadioButton(QString("-y"));
	if (axePrec==QPoint(1,0)) radioX->setChecked(true);
	else if (axePrec==QPoint(0,1)) radioY->setChecked(true);
	else if (axePrec==QPoint(-1,0)) radioMX->setChecked(true);
	else if (axePrec==QPoint(0,-1)) radioMY->setChecked(true);
	else radioX->setChecked(true);
	connect(radioX, SIGNAL(clicked()), this, SLOT(axeDirClicked()));
	connect(radioY, SIGNAL(clicked()), this, SLOT(axeDirClicked()));
	connect(radioMX, SIGNAL(clicked()), this, SLOT(axeDirClicked()));
	connect(radioMY, SIGNAL(clicked()), this, SLOT(axeDirClicked()));

	QHBoxLayout *axeDirLayout = new QHBoxLayout;
	axeDirLayout->addWidget(axeLabel);
	axeDirLayout->addWidget(radioX);
	axeDirLayout->addWidget(radioY);
	axeDirLayout->addWidget(radioMX);
	axeDirLayout->addWidget(radioMY);

	QGroupBox* axeDirBox = new QGroupBox;
	axeDirBox->setFlat(true);
	axeDirBox->setAlignment(Qt::AlignLeft);
	axeDirBox->setLayout(axeDirLayout);	

	//mise en page
	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(directionLabel);
	mainLayout->addWidget(imageDirBox);
	mainLayout->addWidget(segmentDirBox);
	mainLayout->addWidget(axeDirBox);
	mainLayout->addStretch(1);

	mainBox = new QGroupBox(this);
	mainBox->setLayout(mainLayout);
}
DirectionWidget::~DirectionWidget() {
	if (paintInterfSegment!=0) delete paintInterfSegment;
}

void DirectionWidget::imageDirClicked() {
	QString img1 = imageDirCombo1->itemText(imageDirCombo1->currentIndex());
	QString img2 = (nbList==2) ? imageDirCombo2->itemText(imageDirCombo2->currentIndex()) : img1;
	QString img1c = dir+paramMain->convertTifName2Couleur(img1);
	QString img2c = dir+paramMain->convertTifName2Couleur(img2);

	QVector<QPoint> P(2,QPoint(-1,-1));
	for (int i=0; i<2; i++) {
		bool b = !pointsEdit.at(2*i)->text().isEmpty() && !pointsEdit.at(2*i+1)->text().isEmpty();
		if (b) P[i] = QPoint(pointsEdit.at(2*i)->text().toInt(),pointsEdit.at(2*i+1)->text().toInt());
	}

	for (int i=0; i<2; i++) {
		if (paintInterfSegment!=0) delete paintInterfSegment;
		paintInterfSegment = new PaintInterfSegment(paramMain, assistant, pair<QString,QString>(img1c,img2c), this, true, P[0], P[1]);
		if (!paintInterfSegment->getDone()) return;
		paintInterfSegment->show();
	}

	if (paintInterfSegment->exec()!=QDialog::Accepted) return;

	P.fill(QPoint(-1,-1));
	if (paintInterfSegment->getNbPoint(2)>0) P[0] = paintInterfSegment->getSegment().first;
	if ( ((img1==img2) && paintInterfSegment->getNbPoint(2)==2) || (img1!=img2 && paintInterfSegment->getNbPoint(1)==1) )
			P[1] = paintInterfSegment->getSegment().second;
	for (int i=0; i<2; i++) {
		if (P[i]==QPoint(-1,-1)) {
			pointsEdit[2*i]->clear();
			pointsEdit[2*i+1]->clear();
		} else {
			pointsEdit[2*i]->setText(QVariant(P.at(i).x()).toString());
			pointsEdit[2*i+1]->setText(QVariant(P.at(i).y()).toString());
		}
	}
	emit updateParam();
}

void DirectionWidget::axeDirClicked() { emit updateParam(); }

QGroupBox* DirectionWidget::getBox() { return mainBox; }

void DirectionWidget::updateParam(ParamApero* parametres) {
	if (parametres->getUserOrientation().getOrientMethode()!=1) return;
	//images
	parametres->modifUserOrientation().setImage1( imageDirCombo1->itemText(imageDirCombo1->currentIndex()) );
	parametres->modifUserOrientation().setImage2( imageDirCombo2->itemText(imageDirCombo2->currentIndex()) );

	//points
	QVector<QPoint> V(2);
	for (int i=0; i<2; i++) {
		if (!pointsEdit[2*i]->text().isEmpty()) V[i].setX( pointsEdit[2*i]->text().toInt() );
		if (!pointsEdit[2*i+1]->text().isEmpty()) V[i].setY( pointsEdit[2*i+1]->text().toInt() );
	}
	parametres->modifUserOrientation().setPoint1(V[0]);
	parametres->modifUserOrientation().setPoint2(V[1]);

	//axe
	if (radioX->isChecked()) parametres->modifUserOrientation().setAxe(QPoint(1,0));
	else if (radioMX->isChecked()) parametres->modifUserOrientation().setAxe(QPoint(-1,0));
	else if (radioY->isChecked()) parametres->modifUserOrientation().setAxe(QPoint(0,1));
	else parametres->modifUserOrientation().setAxe(QPoint(0,-1));
}
void DirectionWidget::updateParam(CarteDeProfondeur* parametres) {
	//images
	parametres->setImgRep(imageDirCombo1->itemText(imageDirCombo1->currentIndex()));

	//points
	QVector<QPoint> V(2);
	for (int i=0; i<2; i++) {
		if (!pointsEdit[2*i]->text().isEmpty()) V[i].setX( pointsEdit[2*i]->text().toInt() );
		if (!pointsEdit[2*i+1]->text().isEmpty()) V[i].setY( pointsEdit[2*i+1]->text().toInt() );
	}
	parametres->setSegmentRep(pair<QPoint,QPoint>( V[0],V[1] ));

	//axe
	if (radioX->isChecked()) parametres->setAxeRep(QPoint(1,0));
	else if (radioMX->isChecked()) parametres->setAxeRep(QPoint(-1,0));
	else if (radioY->isChecked()) parametres->setAxeRep(QPoint(0,1));
	else parametres->setAxeRep(QPoint(0,-1));
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


EchelleWidget::EchelleWidget(const ParamMain* pMain, int N, const QStringList & liste1, const QStringList & liste2, Assistant* help, const pair<QVector<QString>,QVector<QPoint> > & paramPrec, double mesPrec):
	QWidget(),
	imageEchCombo( QVector<QComboBox*>(N,0) ),
	pointsEdit( QVector<QLineEdit*>(8,0) ),
	paintInterfSegment( QVector<PaintInterfSegment*>( 2, 0 ) ),
	dir( pMain->getDossier() ),
	paramMain( pMain ),
	assistant( help ),
	nbList( N )
{
	QFont font;
	font.setBold(true);
	QLabel* echelleLabel = new QLabel(tr("Scale"));
	echelleLabel->setFont(font);
	//nbList images
	QLabel* imageEchLabel = new QLabel(tr("Images onto draw axis :"));
	for (int i=0; i<nbList; i++) {
		imageEchCombo[i] = new QComboBox;
		imageEchCombo[i]->addItems( (nbList==2 && i==1)? liste2 : liste1);
		if (paramPrec.first.count()>0 && !paramPrec.first.at(i*4/nbList).isEmpty())
			imageEchCombo[i]->setCurrentIndex(imageEchCombo[i]->findText(paramPrec.first.at(i*4/nbList)));
		else
			imageEchCombo[i]->setCurrentIndex(0);
		if (nbList==4) connect(imageEchCombo[i], SIGNAL(currentIndexChanged(QString)), this, SIGNAL(updateParam()));
	}

	imageEchButton = new QPushButton(QIcon(g_iconDirectory+"linguist-check-on.png"), QString());
	imageEchButton->setToolTip(tr("Open these images"));
	imageEchButton->setMaximumSize (QSize(21,21));
	connect(imageEchButton, SIGNAL(clicked()), this, SLOT(imageEchClicked()));

	QGridLayout *imageEchLayout = new QGridLayout;
	imageEchLayout->addWidget(imageEchLabel,0,0,1,1);
	for (int i=0; i<nbList; i++)
		imageEchLayout->addWidget(imageEchCombo[i],1+i/2,i%2,1,1);
	imageEchLayout->addWidget(imageEchButton,1,2,1,2);

	QGroupBox* imageEchBox = new QGroupBox;
	imageEchBox->setFlat(true);
	imageEchBox->setAlignment(Qt::AlignLeft);
	imageEchBox->setLayout(imageEchLayout);

	//4 points
	QLabel* points1Label = new QLabel(tr("segment 1"));
	QLabel* points2Label = new QLabel(tr("segment 2"));
	for (int i=0; i<8; i++) {
		pointsEdit[i] = new QLineEdit;
		pointsEdit[i]->setMaximumWidth(100);
		pointsEdit[i]->setEnabled(false);
	}	
	if (paramPrec.second.count()>0) {
		for (int i=0; i<4; i++) {
			if (paramPrec.second.at(i*4/nbList)==QPoint(-1,-1)) continue;
			pointsEdit[2*i]->setText(QVariant(paramPrec.second.at(i*4/nbList).x()).toString());
			pointsEdit[2*i+1]->setText(QVariant(paramPrec.second.at(i*4/nbList).y()).toString());
		}
	}

	QGridLayout *segmentEchLayout = new QGridLayout;
	segmentEchLayout->addWidget(echelleLabel,0,0,1,1);
	segmentEchLayout->addWidget(points1Label,1,0,1,1);
	for (int i=0; i<4; i++)
		segmentEchLayout->addWidget(pointsEdit[i],1,i+1,1,1);
	segmentEchLayout->addWidget(echelleLabel,2,0,1,1);
	segmentEchLayout->addWidget(points2Label,3,0,1,1);
	for (int i=0; i<4; i++)
		segmentEchLayout->addWidget(pointsEdit[i+4],3,i+1,1,1);

	QGroupBox* segmentEchBox = new QGroupBox;
	segmentEchBox->setFlat(true);
	segmentEchBox->setAlignment(Qt::AlignLeft);
	segmentEchBox->setLayout(segmentEchLayout);

	//distance
	QGroupBox* distEchBox;
	QLabel* distLabel;
	if (nbList==4) distLabel = new QLabel(conv(tr("Real distance")));
	else distLabel = new QLabel(conv(tr("or relative scale :")));
	distEdit = new QLineEdit;
	if (mesPrec!=0)
		distEdit->setText(QVariant(mesPrec).toString());
	connect(distEdit, SIGNAL(textChanged(QString)), this, SIGNAL(updateParam()));

	QHBoxLayout *distEchLayout = new QHBoxLayout;
	distEchLayout->addWidget(distLabel);
	distEchLayout->addWidget(distEdit);
	distEchLayout->addStretch(1);

	distEchBox = new QGroupBox;
	distEchBox->setFlat(true);
	distEchBox->setAlignment(Qt::AlignLeft);
	distEchBox->setLayout(distEchLayout);

	//mise en page
	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(echelleLabel);
	mainLayout->addWidget(imageEchBox);
	mainLayout->addWidget(segmentEchBox);
	mainLayout->addWidget(distEchBox);
	mainLayout->addStretch(1);

	mainBox = new QGroupBox(this);
	mainBox->setLayout(mainLayout);
}
EchelleWidget::~EchelleWidget() {
	for (int i=0; i<2; i++)
		if (paintInterfSegment.at(i)!=0) delete paintInterfSegment[i];
	for (int i=0; i<nbList; i++)
		delete imageEchCombo[i];
	for (int i=0; i<8; i++)
		delete pointsEdit[i];
}

void EchelleWidget::imageEchClicked() {
	QVector<QString> images(nbList);
	QVector<QString> images2(nbList);
	for (int i=0; i<nbList; i++) {
		images[i] = imageEchCombo.at(i)->itemText(imageEchCombo.at(i)->currentIndex());
		images2[i] = dir+paramMain->convertTifName2Couleur(images.at(i));
	}
	if (nbList==4 && (images[0]==images[2] || images[1]==images[3])) {
		qMessageBox(this, tr("Selection error"),conv(tr("The two images on which each point is drawn must be different to recover the third dimension.")));
		return;
	} //si nbList=2, c'est juste un rapport de taille ortho/MNT à calculer (au pire ça sert à rien)

	QVector<QPoint> P(4);
	for (int i=0; i<4; i++) {
		bool b = !pointsEdit.at(2*i)->text().isEmpty() && !pointsEdit.at(2*i+1)->text().isEmpty();
		if (b) P[i] = QPoint(pointsEdit.at(2*i)->text().toInt(),pointsEdit.at(2*i+1)->text().toInt());
		else P[i] = QPoint(-1,-1);
	}

	for (int i=0; i<2; i++) {
		if (paintInterfSegment.at(i)!=0) delete paintInterfSegment[i];
		paintInterfSegment[i] = new PaintInterfSegment(paramMain, assistant, pair<QString,QString>(images2.at(i*nbList/2),images2.at(i*nbList/2+nbList/4)), this, false, P.at(2*i), P.at(2*i+1));
		if (!paintInterfSegment.at(i)->getDone()) return;
		paintInterfSegment[i]->show();
	}

	if (paintInterfSegment[1]->exec()!=QDialog::Accepted || paintInterfSegment[0]->exec()!=QDialog::Accepted) return;

	if (nbList==2) distEdit->clear();
	QVector<QPoint> P2(4,QPoint(-1,-1));
	for (int i=0; i<2; i++) {
		if (paintInterfSegment[i]->getNbPoint(2)>0) P2[2*i] = paintInterfSegment[i]->getSegment().first;
		if (nbList==4 && (((images.at(2*i)==images.at(2*i+1)) && paintInterfSegment[i]->getNbPoint(2)==2) || (images.at(2*i)!=images.at(2*i+1) && paintInterfSegment[i]->getNbPoint(1)==1)) )
			P2[2*i+1] = paintInterfSegment[i]->getSegment().second;
		else if (nbList==2 && paintInterfSegment[i]->getNbPoint(2)==2)
			P2[2*i+1] = paintInterfSegment[i]->getSegment().second;
	}
	for (int i=0; i<4; i++) {
		if (P2[i]==QPoint(-1,-1)) {
			pointsEdit[2*i]->clear();
			pointsEdit[2*i+1]->clear();
		} else {
			pointsEdit[2*i]->setText(QVariant(P2.at(i).x()).toString());
			pointsEdit[2*i+1]->setText(QVariant(P2.at(i).y()).toString());
		}
	}
	emit updateParam();
}

QGroupBox* EchelleWidget::getBox() { return mainBox; }

void EchelleWidget::updateListe2(const QStringList& l) {
	QString text = imageEchCombo.at(1)->itemText(imageEchCombo.at(1)->currentIndex());
	imageEchCombo.at(1)->clear();
	imageEchCombo.at(1)->addItems(l);
	int idx = imageEchCombo.at(1)->findText(text);
	imageEchCombo.at(1)->setCurrentIndex(idx);
	if (idx==-1) {
		for (int i=0; i<8; i++)
			pointsEdit[i]->clear();
		emit updateParam();
	}
}

void EchelleWidget::updateParam(ParamApero* parametres) {
	if (parametres->getUserOrientation().getOrientMethode()!=1 && parametres->getUserOrientation().getOrientMethode()!=2) return;
	//images
	QVector<QString> l(4) ;
	for (int i=0; i<4; i++) l[i] = imageEchCombo.at(i)->itemText(imageEchCombo.at(i)->currentIndex());
	parametres->modifUserOrientation().setImages(l);

	//points
	QVector<QPoint> V(4) ;
	for (int i=0; i<4; i++) {
		if (!pointsEdit[2*i]->text().isEmpty()) V[i].setX( pointsEdit[2*i]->text().toInt() );
		if (!pointsEdit[2*i+1]->text().isEmpty()) V[i].setY( pointsEdit[2*i+1]->text().toInt() );
	}
	parametres->modifUserOrientation().setPoints(V);

	//distance
	QString text = distEdit->text();
	if (text.isEmpty()) return;
	bool ok = true;
	double dist = text.toInt(&ok);
	if (!ok || dist==0) {
		cout << conv(tr("Uncorrect distance for rescaling.")).toStdString() << endl;	
		return;	
	}
	parametres->modifUserOrientation().setDistance(dist);
}
void EchelleWidget::updateParam(CarteDeProfondeur* parametres) {
	if (distEdit->text().isEmpty()) {
		QVector<QPoint> P(4,QPoint(-1,-1));
		for (int i=0; i<4; i++) {
			if (!pointsEdit[2*i]->text().isEmpty()) P[i].setX( pointsEdit[2*i]->text().toInt() );
			else return;
			if (!pointsEdit[2*i+1]->text().isEmpty()) P[i].setY( pointsEdit[2*i+1]->text().toInt() );
			else return;
		}
		QVector<QPoint> S(2);
		for (int i=0; i<2; i++) S[i] = P[2*i+1]-P[2*i];
		if (S[0].isNull() || S[1].isNull()) return;
		QVector<double> d(2);
		for (int i=0; i<2; i++) d[i] = S[i].x()*S[i].x() + S[i].y()*S[i].y();
		parametres->setEchelleOrtho( sqrt(d[0]/d[1]) );
		disconnect(distEdit, SIGNAL(textChanged(QString)), this, SIGNAL(updateParam()));
		distEdit->setText(QVariant(parametres->getEchelleOrtho()).toString());
		connect(distEdit, SIGNAL(textChanged(QString)), this, SIGNAL(updateParam()));
	} else {
		bool ok;
		double ech = distEdit->text().toDouble(&ok);
		if (!ok) {
			cout << conv( tr("Orthoimage scale is unvalid.") ).toStdString() << endl;	
			return;	
		}
		parametres->setEchelleOrtho(ech);
		for (int i=0; i<8; i++) pointsEdit[i]->clear();
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ParamApero::ParamApero():
	imgToOri( QStringList() ),
	imgMaitresse( QString() ),
	userOrientation( UserOrientation() ),
	useOriInit( false ),
	dirOriInit( QString() ),
	autoCalib( QStringList() ),
	multiechelle( false ),
	calibFigees( QList<int>() ),
	liberCalib( QVector<bool>() ),
	filtrage( true ),
	calcPts3D( true ),
	exportPts3D( false ){}

ParamApero::ParamApero(const ParamApero& paramApero) { copie(paramApero); }

ParamApero::~ParamApero () {}

ParamApero& ParamApero::operator=(const ParamApero& paramApero) {
	if (&paramApero!=this) {
		copie(paramApero);
	}
	return *this;
}

void ParamApero::copie(const ParamApero& paramApero) {
	imgToOri =paramApero.getImgToOri();
	imgMaitresse = paramApero.getImgMaitresse();
	userOrientation =paramApero.getUserOrientation();
	useOriInit =paramApero.getUseOriInit();
	dirOriInit =paramApero.getDirOriInit();
	autoCalib = paramApero.getAutoCalib();
	multiechelle = paramApero.getMultiechelle();
	calibFigees =paramApero.getCalibFigees();
	liberCalib = paramApero.getLiberCalib();
	filtrage = paramApero.getFiltrage();
	calcPts3D = paramApero.getCalcPts3D();
	exportPts3D = paramApero.getExportPts3D();
}

const QStringList& ParamApero::getImgToOri() const { return imgToOri; }
QStringList& ParamApero::modifImgToOri() { return imgToOri; }
const QString& ParamApero::getImgMaitresse() const { return imgMaitresse; }
const UserOrientation& ParamApero::getUserOrientation() const { return userOrientation; }
UserOrientation& ParamApero::modifUserOrientation() { return userOrientation; }
bool ParamApero::getUseOriInit() const { return useOriInit; }
const QString& ParamApero::getDirOriInit() const { return dirOriInit; }
const QStringList& ParamApero::getAutoCalib() const { return autoCalib; }
QStringList& ParamApero::modifAutoCalib() { return autoCalib; }
bool ParamApero::getMultiechelle() const { return multiechelle; }
const QList<int>& ParamApero::getCalibFigees() const { return calibFigees; }
QList<int>& ParamApero::modifCalibFigees() { return calibFigees; }
const QVector<bool>& ParamApero::getLiberCalib() const { return liberCalib; }
QVector<bool>& ParamApero::modifLiberCalib() { return liberCalib; }
bool ParamApero::getFiltrage() const { return filtrage; }
bool ParamApero::getCalcPts3D() const { return calcPts3D; }
bool ParamApero::getExportPts3D() const { return exportPts3D; }

void ParamApero::setImgToOri(const QStringList& l) { imgToOri = l; }
void ParamApero::setImgMaitresse(const QString& s) { imgMaitresse = s; }
void ParamApero::setUserOrientation(const UserOrientation& uo) { userOrientation = uo; }
void ParamApero::setUseOriInit(bool u) { useOriInit = u; }
void ParamApero::setDirOriInit(const QString& d) { dirOriInit = d; }
void ParamApero::setAutoCalib(const QStringList& l) { autoCalib = l; }
void ParamApero::setMultiechelle(bool b) { multiechelle = b; }
void ParamApero::setCalibFigees(const QList<int>& cf) { calibFigees = cf; }
void ParamApero::setLiberCalib(const QVector<bool>& lc) { liberCalib = lc; }
void ParamApero::setFiltrage(bool b) { filtrage = b; }
void ParamApero::setCalcPts3D(bool b) { calcPts3D = b; }
void ParamApero::setExportPts3D(bool b) { exportPts3D = b; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


UserOrientation::UserOrientation() : orientMethode(0), bascOnPlan(false), imgMasque(QString()), image1(QString()), image2(QString()), point1(QPoint(-1,-1)), point2(QPoint(-1,-1)), axe(QPoint(1,0)), fixEchelle(false), points(QVector<QPoint>(4,QPoint(-1,-1))), images(QVector<QString>(4)), distance(1), imageGeoref(QString()), georefFile(QString()), centreAbs(QVector<REAL>(3,0)), rotationAbs(QVector<REAL>(9,0)), pointsGPS(QString()), appuisImg(QString()) {}

UserOrientation::UserOrientation(const UserOrientation& userOrientation) {
	copie(userOrientation);
}

UserOrientation::~UserOrientation () {}

UserOrientation& UserOrientation::operator=(const UserOrientation& userOrientation) {
	if (&userOrientation!=this)
		copie(userOrientation);
	return *this;
}

void UserOrientation::copie(const UserOrientation& userOrientation) {
	orientMethode = userOrientation.orientMethode;
	bascOnPlan = userOrientation.getBascOnPlan();
	imgMasque = userOrientation.imgMasque;
	image1 = userOrientation.image1;
	image2 = userOrientation.image2;
	point1 = userOrientation.point1;
	point2 = userOrientation.point2;
	axe = userOrientation.getAxe();
	fixEchelle = userOrientation.getFixEchelle();
	points = userOrientation.points;
	images = userOrientation.images;
	distance = userOrientation.distance;
	imageGeoref = userOrientation.imageGeoref;
	georefFile = userOrientation.georefFile;
	centreAbs = userOrientation.centreAbs;
	rotationAbs = userOrientation.rotationAbs;
	pointsGPS = userOrientation.getPointsGPS();
	appuisImg = userOrientation.getAppuisImg();
}

int UserOrientation::getOrientMethode() const { return orientMethode; }
bool UserOrientation::getBascOnPlan() const { return bascOnPlan; }
const QString& UserOrientation::getImgMasque() const { return imgMasque; }
const QString& UserOrientation::getImage1() const { return image1; }
const QString& UserOrientation::getImage2() const { return image2; }
const QPoint& UserOrientation::getPoint1() const { return point1; }
const QPoint& UserOrientation::getPoint2() const { return point2; }
const QPoint& UserOrientation::getAxe() const { return axe; }
bool UserOrientation::getFixEchelle() const { return fixEchelle; }
const QVector<QPoint>& UserOrientation::getPoints() const { return points; }
QVector<QPoint>& UserOrientation::modifPoints() { return points; }
const QVector<QString>& UserOrientation::getImages() const { return images; }
QVector<QString>& UserOrientation::modifImages() { return images; }
double UserOrientation::getDistance() const { return distance; }
const QString& UserOrientation::getImageGeoref() const { return imageGeoref; }
const QString& UserOrientation::getGeorefFile() const { return georefFile; }
const QVector<REAL>& UserOrientation::getCentreAbs() const { return centreAbs; }
const QVector<REAL>& UserOrientation::getRotationAbs() const { return rotationAbs; }
const QString& UserOrientation::getPointsGPS() const { return pointsGPS; }
const QString& UserOrientation::getAppuisImg() const { return appuisImg; }

void UserOrientation::setOrientMethode(int om) { orientMethode = om; }
void UserOrientation::setBascOnPlan(bool b) { bascOnPlan = b; }
void UserOrientation::setImgMasque(const QString& m) { imgMasque = m; }
void UserOrientation::setImage1(const QString& i) { image1 = i; }
void UserOrientation::setImage2(const QString& i) { image2 = i; }
void UserOrientation::setPoint1(const QPoint& p) { point1 = p; }
void UserOrientation::setPoint2(const QPoint& p) { point2 = p; }
void UserOrientation::setAxe(const QPoint& p) { axe = p; }
void UserOrientation::setFixEchelle(bool f) { fixEchelle = f; }
void UserOrientation::setPoints(const QVector<QPoint>& l) { points = l; }
void UserOrientation::setImages(const QVector<QString>& l) { images = l; }
void UserOrientation::setDistance(double d) { distance = d; }
void UserOrientation::setImageGeoref(const QString& i) { imageGeoref = i; }
void UserOrientation::setGeorefFile(const QString& g) { georefFile = g; }
void UserOrientation::setCentreAbs(const QVector<REAL>& c) { centreAbs = c; }
void UserOrientation::setRotationAbs(const QVector<REAL>& r) { rotationAbs = r; }
QVector<REAL>& UserOrientation::modifCentreAbs() { return centreAbs; }
QVector<REAL>& UserOrientation::modifRotationAbs() { return rotationAbs; }
void UserOrientation::setPointsGPS(const QString& file) { pointsGPS = file; }
void UserOrientation::setAppuisImg(const QString& file) { appuisImg = file; }

const QString UserOrientation::getMasque() const { return imgMasque.section(".",0,-2)+QString("_MasqPlan.tif"); }
const QString UserOrientation::getRefMasque() const { return getMasque().section(".",0,-2)+QString(".xml"); }

