#include "interfMicmac.h"

using namespace std;


QString imgNontuilee(const QString& img) { return img.section(".",0,-2) + QString("nontuile.tif"); }


InterfMicmac::InterfMicmac(Interface* parent, const ParamMain* param, int typChan, VueChantier* vue, Assistant* help) : 
	QDialog( parent ),
	cartesTab( 0 ),
	mNTTab( 0 ),
	repereTab( 0 ),
	maskTab( 0 ),
	orthoTab( 0 ),
	profondeurTab( 0 ),
	assistant( help ),
	paramMain( param ),
	paramMicmac( param->getParamMicmac() ),
	typeChantier( typChan ),
	dir( param->getDossier() ),
	carteCourante( 0 ),
	vueChantier( vue )
{
	//setWindowModality(Qt::ApplicationModal);
	setMaximumSize(maximumSizeHint());

	//tabs
	tabWidget = new QTabWidget;
	tabWidget->setMovable (false);

	cartesTab = new CartesTab (this, paramMain, &(paramMicmac));
	tabWidget->addTab(cartesTab, tr("Depth maps"));
	connect(cartesTab, SIGNAL(modifCarte(CarteDeProfondeur*)), this, SLOT(modifCarte(CarteDeProfondeur*)));
	connect(cartesTab, SIGNAL(updateCalcButton()), this, SLOT(updateCalcButton()));
	connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updateInterface()));

	//toolbar
	QDialogButtonBox* buttonBox = new QDialogButtonBox();

	QPushButton* helpButton = new QPushButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	helpButton->setToolTip(tr("Help"));
	buttonBox->addButton (helpButton, QDialogButtonBox::HelpRole);
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));

	QPushButton* cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::ActionRole);
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(cancelClicked()));

	precButton = buttonBox->addButton (QApplication::translate("Dialog", tr("Previous").toStdString().c_str(), 0, QApplication::CodecForTr), QDialogButtonBox::AcceptRole);
	precButton->setToolTip(tr("Previous tab"));
	connect(precButton, SIGNAL(clicked()), this, SLOT(precClicked()));

	suivantButton = buttonBox->addButton (tr("Next"), QDialogButtonBox::ActionRole);
	suivantButton->setToolTip(tr("Next tab"));
	connect(suivantButton, SIGNAL(clicked()), this, SLOT(suivClicked()));

	saveButton = buttonBox->addButton (tr("Ok"), QDialogButtonBox::ActionRole);
	saveButton->setToolTip(conv(tr("Save this depth map parameters")));
	connect(saveButton, SIGNAL(clicked()), this, SLOT(saveClicked()));

	calButton = buttonBox->addButton (tr("Compute"), QDialogButtonBox::AcceptRole);
	calButton->setToolTip(tr("Launch all depth map computation"));
	connect(calButton, SIGNAL(clicked()), this, SLOT(calcClicked()));

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(tabWidget);
	mainLayout->addWidget(buttonBox);
	setLayout(mainLayout);
	setWindowTitle(tr("Depth map computing"));
	layout()->setSizeConstraint(QLayout::SetFixedSize);
	updateInterface();
}
InterfMicmac::~InterfMicmac() {
	if (cartesTab!=0) delete cartesTab;
	if (mNTTab!=0) delete mNTTab;
	if (repereTab!=0) delete repereTab;
	if (maskTab!=0) delete maskTab;
	if (orthoTab!=0) delete orthoTab;
	if (profondeurTab!=0) delete profondeurTab;
}

QSize InterfMicmac::maximumSizeHint() const { return QApplication::desktop()->availableGeometry().size()-QSize(0,100); }

void InterfMicmac::modifCarte(CarteDeProfondeur* carte) {
	carteCourante = carte;
	saveButton->setEnabled(false);
	mNTTab = new MNTTab (this, paramMain, carteCourante, vueChantier, &(paramMicmac), assistant);
	connect(mNTTab, SIGNAL(suiteMNT(bool)), this, SLOT(suiteMNT(bool)));
	tabWidget->addTab(mNTTab, conv(tr("Images for correlation")));
	if (!carteCourante->getImageDeReference().isEmpty() && carteCourante->getImagesCorrel().count()>0) suiteMNT(true);
	tabWidget->setCurrentWidget(mNTTab);
	cartesTab->setEnabled(false);
	calButton->setEnabled(false);
}

void InterfMicmac::updateCalcButton() { calButton->setEnabled(cartesACalculer()); }

void InterfMicmac::suiteMNT(bool b) {
	disconnect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updateInterface()));
	if (b) {
		if (repereTab==0) {
			repereTab = new RepereTab (this, paramMain, vueChantier, carteCourante, assistant);
			connect(repereTab, SIGNAL(suiteRepere(bool)), this, SLOT(suiteRepere(bool)));
			tabWidget->addTab(repereTab, conv(tr("Correlation frame")));
	//	if (QFile(carteCourante->getImageSaisie(*paramMain)).exists() && (!carteCourante->getAutreRepere() || QFile(carteCourante->getRepereFile(*paramMain)).exists())) suiteRepere(true);
		}
	} else {
		if (repereTab!=0) delete repereTab;
		if (maskTab!=0) delete maskTab;
		if (orthoTab!=0) delete orthoTab;
		if (profondeurTab!=0) delete profondeurTab;
		repereTab = 0;
		maskTab = 0;
		orthoTab = 0;
		profondeurTab = 0;
	}
	connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updateInterface()));
}

void InterfMicmac::suiteRepere(bool b) {
	disconnect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updateInterface()));
	if (b) {
			if (maskTab!=0) delete maskTab;
			maskTab = new MaskTab (this, paramMain, carteCourante, assistant);
			connect(maskTab, SIGNAL(suiteMasque()), this, SLOT(suite()));
			if (orthoTab==0 && profondeurTab==0) tabWidget->addTab(maskTab, tr("Mask"));
			else tabWidget->insertTab(3,maskTab, tr("Mask"));
		if (orthoTab==0 && !carteCourante->getRepere()) {
			orthoTab = new OrthoTab (this, paramMain, carteCourante, assistant);
			connect(orthoTab, SIGNAL(suiteOrtho()), this, SLOT(suite()));
			tabWidget->addTab(orthoTab, tr("Orthoimages"));
		}
		if (profondeurTab==0) {
			profondeurTab = new ProfondeurTab (this, paramMain, carteCourante);
			connect(profondeurTab, SIGNAL(suiteProf()), this, SLOT(suite()));
			tabWidget->addTab(profondeurTab, tr("Relief"));
		}
		suite();
	} else {
		if (maskTab!=0) delete maskTab;
		if (orthoTab!=0) delete orthoTab;
		if (profondeurTab!=0) delete profondeurTab;
		maskTab = 0;
		orthoTab = 0;
		profondeurTab = 0;
	}
	connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updateInterface()));
}

void InterfMicmac::suite() {
cout << (!carteCourante->getImageDeReference().isEmpty()) << " && " << (carteCourante->getImagesCorrel().count()>0) << endl;
cout << (QFile(carteCourante->getImageSaisie(*paramMain)).exists()) << " && " << (!carteCourante->getAutreRepere()) << " || " << (QFile(dir+carteCourante->getRepereFile(*paramMain)).exists()) << endl;
cout << (QFile(carteCourante->getMasque(*paramMain)).exists()) << " && " << (QFile(carteCourante->getReferencementMasque(*paramMain)).exists()) << endl;
cout << (carteCourante->getRepere()) << " || " << (!carteCourante->getDoOrtho()) << " || " << (carteCourante->getImgsOrtho().count()>0) << " && " << (carteCourante->getEchelleOrtho()>0) << endl;
cout << (!carteCourante->getImageDeReference().isEmpty()) << " " << (carteCourante->getImagesCorrel().count()>0) << endl;
cout << (carteCourante->getInterv().first>0) << " && " << (carteCourante->getInterv().second>carteCourante->getInterv().first) << " && " << (!carteCourante->getDiscontinuites()) << " || " << (carteCourante->getSeuilZRelatif()>0) << endl;
cout << carteCourante->getRepereFile(*paramMain).toStdString() << endl;
cout << carteCourante->getReferencementMasque(*paramMain).toStdString() << endl;
	if (	!carteCourante->getImageDeReference().isEmpty() && carteCourante->getImagesCorrel().count()>0
		&& QFile(carteCourante->getImageSaisie(*paramMain)).exists() && (!carteCourante->getAutreRepere() || QFile(dir+carteCourante->getRepereFile(*paramMain)).exists())
		&& QFile(carteCourante->getMasque(*paramMain)).exists() && QFile(carteCourante->getReferencementMasque(*paramMain)).exists()
		&& (carteCourante->getRepere() || !carteCourante->getDoOrtho() || (carteCourante->getImgsOrtho().count()>0 && carteCourante->getEchelleOrtho()>0) )
		&& carteCourante->getInterv().first>0 && carteCourante->getInterv().second>carteCourante->getInterv().first && (!carteCourante->getDiscontinuites() || carteCourante->getSeuilZRelatif()>0)
	)
		saveButton->setEnabled(true);
	else
		saveButton->setEnabled(false);
}

void InterfMicmac::updateInterface() {
	int tab = tabWidget->currentIndex();
	int N = tabWidget->count();
	precButton->hide();
	suivantButton->hide();
	saveButton->hide();
	calButton->hide();

	if (N==1) {
		calButton->show();
		cartesTab->enableSelect(true);
	} else {
		if (tab!=0) precButton->show();
		if (tab!=N-1) suivantButton->show();
		if ((N==6 && !carteCourante->getRepere()) || (N==5 && carteCourante->getRepere())) saveButton->show();
		cartesTab->enableSelect(false);
	}
}
void InterfMicmac::precClicked() { tabWidget->setCurrentIndex(tabWidget->currentIndex()-1); }
void InterfMicmac::suivClicked() { tabWidget->setCurrentIndex(tabWidget->currentIndex()+1); }

void InterfMicmac::cancelClicked() {
	if (tabWidget->count()==1) reject();
	else {
		for (int i=1; i<tabWidget->count(); i++) tabWidget->removeTab(i);
		carteCourante = 0;
		if (mNTTab!=0) mNTTab->close();
		if (repereTab!=0) delete repereTab;
		if (maskTab!=0) delete maskTab;
		if (orthoTab!=0) delete orthoTab;
		if (profondeurTab!=0) delete profondeurTab;
		mNTTab = 0;
		repereTab = 0;
		maskTab = 0;
		orthoTab = 0;
		profondeurTab = 0;
		cartesTab->setEnabled(true);
		calButton->setEnabled(cartesACalculer());
	}
}

void InterfMicmac::saveClicked() {
//sauvegarde des paramètres de la carte en cours
	if (carteCourante->getRepere()) carteCourante->setDoOrtho(false);

	int idx = cartesTab->getCarte(paramMicmac,carteCourante->getImageDeReference());
	if (idx==paramMicmac.count()) paramMicmac.push_back(*carteCourante);
	else paramMicmac[idx] = *carteCourante;

	for (int i=1; i<tabWidget->count(); i++) tabWidget->removeTab(i);
	if (mNTTab!=0) mNTTab->close();
	if (repereTab!=0) delete repereTab;
	if (maskTab!=0) delete maskTab;
	if (orthoTab!=0) delete orthoTab;
	if (profondeurTab!=0) delete profondeurTab;
	mNTTab = 0;
	repereTab = 0;
	maskTab = 0;
	orthoTab = 0;
	profondeurTab = 0;
	carteCourante = 0;
	cartesTab->updateListe();
	cartesTab->setEnabled(true);
	calButton->setEnabled(cartesACalculer());
}

void InterfMicmac::calcClicked() {
	hide();
	if (cartesACalculer()) accept();
	else reject();
}

void InterfMicmac::helpClicked() {
	if (tabWidget->currentWidget()==cartesTab) assistant->showDocumentation(assistant->pageInterfMicmac);
	else if (tabWidget->currentWidget()==mNTTab) assistant->showDocumentation(assistant->pageInterfMicmacMNT);
	else if (tabWidget->currentWidget()==repereTab) assistant->showDocumentation(assistant->pageInterfMicmacRepere);
	else if (tabWidget->currentWidget()==maskTab) assistant->showDocumentation(assistant->pageInterfMicmacMasque);
	else if (tabWidget->currentWidget()==orthoTab) assistant->showDocumentation(assistant->pageInterfMicmacOrtho);
	else if (tabWidget->currentWidget()==profondeurTab) assistant->showDocumentation(assistant->pageInterfMicmacProfondeur);
}

bool InterfMicmac::cartesACalculer() {
	for (int i=0; i<paramMicmac.count(); i++) {
		if (paramMicmac.at(i).getACalculer())
			return true;
	}
	return false;
}

const QVector<CarteDeProfondeur>& InterfMicmac::getParamMicmac() const { return paramMicmac; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


CartesTab::CartesTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  QVector<CarteDeProfondeur>* param):
	QScrollArea(),
	resizableWidget( 0 ),
	parametres( param ),
	parent( interfMicmac ),
	paramMain( pMain )
{
	//liste des cartes créées et importées
	treeWidget = new QTreeWidget;
	treeWidget->setColumnCount(3);
	treeWidget->setSelectionMode (QAbstractItemView::SingleSelection);
	treeWidget->setHeaderLabels(QStringList(conv(tr("Reference images")))
					<< (paramMain->isFrench()? QApplication::translate("Dialog", "Frame", 0, QApplication::CodecForTr) : QString("Frame"))
					<< tr("Orthoimages"));
	treeWidget->setPalette(QPalette(QColor(255,255,255)));
	if (parametres->count()>0) {
		for (QVector<CarteDeProfondeur>::const_iterator it=parametres->begin(); it!=parametres->end(); it++) {
			QStringList l(it->getImageDeReference());
			l.push_back( (it->getRepere())? tr("image") : ((it->getAutreRepere())? it->getRepereFile(*paramMain) : tr("Euclidean")) );
			l.push_back( (!it->getDoOrtho())? QString("X") : QString(QChar(8730)) );
			QTreeWidgetItem* twi = new QTreeWidgetItem(l);
			treeWidget->addTopLevelItem(twi);
		}
	}
	resizeTreeWidget();
	connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));

	//boutons
	addButton = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"), QString());
	addButton->setToolTip(conv(tr("Add new depth map")));
	addButton->setMaximumSize (QSize(32,32));
	addButton->setEnabled(treeWidget->topLevelItemCount()<paramMain->getParamApero().getImgToOri().count());
	connect(addButton, SIGNAL(clicked()), this, SLOT(addNewCarte()));

	removeButton = new QPushButton(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic2.png"), QString());
	removeButton->setToolTip(conv(tr("Remove depth map from computation")));
	removeButton->setMaximumSize (QSize(32,32));
	removeButton->setEnabled(false);
	connect(removeButton, SIGNAL(clicked()), this, SLOT(removeCartes()));

	modifButton = new QPushButton(QIcon(g_iconDirectory+"designer-edit-resources-button.png"), QString());
	modifButton->setToolTip(conv(tr("Modify depth map parameters")));
	modifButton->setMaximumSize (QSize(32,32));
	modifButton->setEnabled(false);
	connect(modifButton, SIGNAL(clicked()), this, SLOT(modifCarte()));

	QHBoxLayout* toolLayout = new QHBoxLayout;
	toolLayout->addWidget(addButton,0,Qt::AlignHCenter);
	toolLayout->addWidget(removeButton,0,Qt::AlignHCenter);
	toolLayout->addWidget(modifButton,0,Qt::AlignHCenter);
	toolLayout->addStretch();
	QGroupBox* toolBox = new QGroupBox;
	toolBox->setFlat(false);
	toolBox->setLayout(toolLayout);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(treeWidget,0,Qt::AlignHCenter);
	mainLayout->addWidget(toolBox,0,Qt::AlignHCenter);
	mainLayout->addStretch();

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);

	ignoreAct = new QAction(tr("&Do not compute again"), this);
	connect(ignoreAct, SIGNAL(triggered()), this, SLOT(ignore()));
}
CartesTab::~CartesTab() {
	delete ignoreAct;
}

void CartesTab::enableSelect(bool b) {
	if (b) treeWidget->setSelectionMode (QAbstractItemView::SingleSelection);
	else treeWidget->setSelectionMode (QAbstractItemView::NoSelection);
}

QSize CartesTab::sizeHint() {
	QSize sizeH = QApplication::desktop()->availableGeometry().size()/2;
	treeWidget->setColumnWidth(0,2*sizeH.width()/5);
	treeWidget->setColumnWidth(1,2*sizeH.width()/5);
	return sizeH;
}

QSize CartesTab::minimumSizeHint() {
	QSize size1 = QApplication::desktop()->availableGeometry().size()+parent->size()-size()-QSize(0,100);
	QSize size2;
	if (resizableWidget!=0) size2 = resizableWidget->size();
	QSize size3;
	if (!size2.isNull()) size3 = QSize( min(size1.width(),size2.width()) , min(size1.height(),size2.height()) );
	else size3 = size1;
	return size3;
}

void CartesTab::resizeEvent(QResizeEvent* event) {
	resizeTreeWidget();
	setMinimumSize(minimumSizeHint());
	setMaximumSize(QApplication::desktop()->availableGeometry().size()+parent->size()-size()-QSize(0,100));
	QWidget::resizeEvent(event);
	treeWidget->setColumnWidth(0,2*(width()-25)/5);
	treeWidget->setColumnWidth(1,2*(width()-25)/5);
}

void CartesTab::resizeTreeWidget() {	
	QFontMetrics metrics = QFontMetrics(treeWidget->font());
	int h = 0;
	for (int k=0; k<3; k++) {
		treeWidget->resizeColumnToContents(k);
		QString text = treeWidget->headerItem()->text(k);
		for (int i=0; i<treeWidget->topLevelItemCount(); i++) {
			text += QString("\n") + treeWidget->topLevelItem(i)->text(k);
		}
		QRect maxr(0,0,QApplication::desktop()->availableGeometry().width()/2,QApplication::desktop()->availableGeometry().height()/2);
		QRect r = metrics.boundingRect(maxr, Qt::AlignLeft|Qt::AlignVCenter, text);
		if (r.height()>h) h = r.height();
	}
	treeWidget->setFixedHeight(h+40);	//ascenseur + en-têtes de treeWidget
	treeWidget->setFixedWidth(treeWidget->columnWidth(0)+treeWidget->columnWidth(1)+treeWidget->columnWidth(2));
}

void CartesTab::contextMenuEvent(QContextMenuEvent *event) {
	if (treeWidget->selectedItems().count()==0) return;
	if (treeWidget->geometry().contains(treeWidget->parentWidget()->mapFrom(this,event->pos())) && treeWidget->topLevelItemCount()>0) {
		int idx = getCarte(*parametres, treeWidget->selectedItems().first()->text(0));
		if (parametres->at(idx).getACalculer()) ignoreAct->setText(tr("&Do not compute again"));
		else ignoreAct->setText(tr("&Compute"));
		QMenu menu(this);
		menu.addAction(ignoreAct);
		menu.exec(event->globalPos());
	}
}

void CartesTab::ignore() {
	QList<QTreeWidgetItem*> l = treeWidget->selectedItems();
	if (l.count()==0) return;
	for (QList<QTreeWidgetItem*>::iterator it=l.begin(); it!=l.end(); it++) {
		for (int j=0; j<3; j++) (*it)->setBackground(j, QBrush(QColor(150,150,150)) );
		int idx = getCarte(*parametres, (*it)->text(0));
		(*parametres)[idx].setACalculer( (ignoreAct->text()==tr("&Compute")) );
	}
	emit updateCalcButton();
}

void CartesTab::selectionChanged() {
	QList<QTreeWidgetItem*> l = treeWidget->selectedItems();
	removeButton->setEnabled(l.count()>0);
	modifButton->setEnabled(l.count()>0);
}

void CartesTab::addNewCarte() {
	carteCourante = CarteDeProfondeur();
	emit modifCarte(&carteCourante);
}

void CartesTab::removeCartes() {
	QList<QTreeWidgetItem*> l = treeWidget->selectedItems();
	if (l.count()==0) return;
	for (QList<QTreeWidgetItem*>::iterator it=l.begin(); it!=l.end(); it++) {
		int idx = getCarte(*parametres, (*it)->text(0));
		treeWidget->removeItemWidget(*it,0);
		parametres->remove(idx);
	}
	updateListe();
	emit updateCalcButton();
}

void CartesTab::modifCarte() {
	QList<QTreeWidgetItem*> l = treeWidget->selectedItems();
	if (l.count()==0) return;
	if (l.count()==0) {
		qMessageBox(this, tr("Error"), conv(tr("Only one depth map can be modified at the same time.")));
		return;
	}
	int idx = getCarte(*parametres, l.first()->text(0));
	carteCourante = (*parametres)[idx];	//copie au cas où cancel
	emit modifCarte(&carteCourante);
}

void CartesTab::updateListe() {
//si une carte a été modifiée ou ajoutée
	treeWidget->clear();
	if (parametres->count()>0) {
		for (QVector<CarteDeProfondeur>::const_iterator it=parametres->begin(); it!=parametres->end(); it++) {
			QStringList l(it->getImageDeReference());
			l.push_back( (it->getRepere())? tr("image") : (it->getAutreRepere())? it->getRepereFile(*paramMain) : tr("Euclidean") );
			l.push_back( (!it->getDoOrtho())? QString("X") : QString(QChar(8730)) );
			QTreeWidgetItem* twi = new QTreeWidgetItem(l);
			treeWidget->addTopLevelItem(twi);
		}
	}
	addButton->setEnabled(treeWidget->topLevelItemCount()<paramMain->getParamApero().getImgToOri().count());
	selectionChanged();
	resizeTreeWidget();
}

int CartesTab::getCarte(const QVector<CarteDeProfondeur>& parametres, const QString& nomImg) const {
//index de la carte d'image de référence nomImg dans parametres ; renvoie parametres.count() si elle n'est pas trouvée
	int idx=0;
	while (idx<parametres.count()) {
		if (nomImg==parametres.at(idx).getImageDeReference()) break;
		else idx++;
	}
	return idx;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MNTTab::MNTTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  CarteDeProfondeur* param, VueChantier* vueChantier, const QVector<CarteDeProfondeur>* cartes, Assistant* help):
	QScrollArea(),
	resizableWidget( 0 ),
	assistant( help ),
	parametre( param ),
	parent( interfMicmac ),
	paramMain( pMain ),
	vue3D( vueChantier )
{
	QStringList l;
	for (QVector<CarteDeProfondeur>::const_iterator it=cartes->begin(); it!=cartes->end(); it++) {
		if (!it->getImageDeReference().isEmpty())
			l.push_back(it->getImageDeReference());
	}

	//choix de l'image de référence
	QLabel* imageLabel = new QLabel(conv(tr("Reference image :")));
	imageCombo = new QComboBox;
	imageCombo->setMinimumWidth(150);
	if (!parametre->getImageDeReference().isEmpty()) imageCombo->addItem(parametre->getImageDeReference());
	for (QStringList::const_iterator it=paramMain->getParamApero().getImgToOri().begin(); it!=paramMain->getParamApero().getImgToOri().end(); it++) {
		if (!l.contains(*it))
			imageCombo->addItem(*it);
	}
	if (!parametre->getImageDeReference().isEmpty()) imageCombo->setCurrentIndex( imageCombo->findText(parametre->getImageDeReference()) );
	else imageCombo->setCurrentIndex(-1);
	connect(imageCombo, SIGNAL(currentIndexChanged(QString)), this, SLOT(imgRefChanged(QString)));

	vue3DButton = new QPushButton(QIcon(g_iconDirectory+"viewmag.png"), QString());
	vue3DButton->setToolTip(conv(tr("Select graphically reference image in the 3D view")));
	vue3DButton->setMaximumSize (QSize(32,32));
	connect(vue3DButton, SIGNAL(clicked()), this, SLOT(vue3DClicked()));

	QHBoxLayout *imageLayout = new QHBoxLayout;
	imageLayout->addWidget(imageLabel);
	imageLayout->addWidget(imageCombo);
	imageLayout->addWidget(vue3DButton);
	imageLayout->addStretch();

	QGroupBox* imageBox = new QGroupBox;
	imageBox->setFlat(true);
	imageBox->setAlignment(Qt::AlignLeft);
	imageBox->setLayout(imageLayout);

	//liste des images pour la corrélation
	QLabel* correlLabel = new QLabel(conv(tr("Images to be used for correlation :")));

	correlImgsList = new QListWidget;
	correlImgsList->setSelectionMode (QAbstractItemView::ExtendedSelection);
	if (parametre->getImagesCorrel().count()>0) correlImgsList->addItems(parametre->getImagesCorrel());
	connect(correlImgsList, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
	
	addCorrelImgButton = new QToolButton;
	addCorrelImgButton->setIcon(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"));
	addCorrelImgButton->setMaximumSize (QSize(32,32));
	addCorrelImgButton->setToolTip(conv(tr("Add an image for correlation")));
	addCorrelImgButton->setEnabled(parametre->getImagesCorrel().count()<paramMain->getParamApero().getImgToOri().count()-1);
	connect(addCorrelImgButton, SIGNAL(clicked()), this, SLOT(addCorrelImgClicked()));

	addFromList = new QAction(conv(tr("Select images in a list")), this);
	connect(addFromList, SIGNAL(triggered()), this, SLOT(addFromListClicked()));
	addFromView = new QAction(conv(tr("Select a camera from 3D view")), this);
	connect(addFromView, SIGNAL(triggered()), this, SLOT(addFromViewClicked()));
	addFromStat = new QAction(conv(tr("Add the best cameras")), this);
	connect(addFromStat, SIGNAL(triggered()), this, SLOT(addFromStatClicked()));
	
	removeCorrelImgButton = new QPushButton;
	removeCorrelImgButton->setIcon(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic2.png"));
	removeCorrelImgButton->setMaximumSize (QSize(32,32));
	removeCorrelImgButton->setToolTip(conv(tr("Remove images for correlation")));
	removeCorrelImgButton->setEnabled(parametre->getImagesCorrel().count()>0);
	connect(removeCorrelImgButton, SIGNAL(clicked()), this, SLOT(removeCorrelImgClicked()));

	QGridLayout *correlLayout = new QGridLayout;
	correlLayout->addWidget(correlLabel,0,0,1,3,Qt::AlignHCenter);
	correlLayout->addWidget(correlImgsList,1,0,Qt::AlignHCenter);
	correlLayout->addWidget(addCorrelImgButton,1,1,Qt::AlignHCenter);
	correlLayout->addWidget(removeCorrelImgButton,1,2,Qt::AlignHCenter);

	QGroupBox* correlBox = new QGroupBox;
	correlBox->setFlat(false);
	correlBox->setLayout(correlLayout);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(imageBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(correlBox,0,Qt::AlignHCenter);
	mainLayout->addStretch();

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
	if (correlImgsList->count()==paramMain->getParamApero().getImgToOri().count()) addCorrelImgButton->setEnabled(false);
}
MNTTab::~MNTTab() {
	delete addFromList;
	delete addFromView;
	delete addFromStat;
}

QSize MNTTab::sizeHint () { return QApplication::desktop()->availableGeometry().size()/2; }

QSize MNTTab::minimumSizeHint () {
	QSize size1 = QApplication::desktop()->availableGeometry().size()+parent->size()-size()-QSize(0,100);
	QSize size2;
	if (resizableWidget!=0) size2 = resizableWidget->size();
	QSize size3;
	if (!size2.isNull()) size3 = QSize( min(size1.width(),size2.width()) , min(size1.height(),size2.height()) );
	else size3 = size1;
	return size3;
}

void MNTTab::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event);
	setMinimumSize(minimumSizeHint());
	setMaximumSize(QApplication::desktop()->availableGeometry().size()+parent->size()-size()-QSize(0,100));
}

void MNTTab::imgRefChanged(QString img) {
	if (img==parametre->getImageDeReference()) return;
	QString img0 = parametre->getImageDeReference();
	parametre->setImageDeReference(img);
	parametre->modifImagesCorrel().clear();
	correlImgsList->clear();
	addCorrelImgButton->setEnabled(true);
	emit suiteMNT(false);
}

void MNTTab::vue3DClicked() {	
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vue3D==0) vue3D = new VueChantier(paramMain, this, assistant);
	vue3D->show(SelectCamBox::RefImage);
	QApplication::restoreOverrideCursor();
	int rep = vue3D->exec();
	if (rep != QDialog::Accepted) return;
        if (vue3D->getRefImg().count()==0) return;
        QString fichier = vue3D->getRefImg().at(0);
	imageCombo->setCurrentIndex(imageCombo->findText(fichier));
	imgRefChanged(fichier);
}

void MNTTab::addCorrelImgClicked() {
	//affiche le menu de addCorrelImgButton : ajout à partir de la liste ou de la vue
	QMenu menu(addCorrelImgButton);
	menu.addAction(addFromList);
	menu.addAction(addFromView);
	menu.addAction(addFromStat);
	menu.exec(addCorrelImgButton->mapToGlobal(QPoint(addCorrelImgButton->width(), 0)));
}

void MNTTab::addFromListClicked() {
	//l'utilisateur doit sélectionner une caméra dans une liste
		//liste des images sélectionnables
	QStringList l;
	for (QStringList::const_iterator it=paramMain->getParamApero().getImgToOri().begin(); it!=paramMain->getParamApero().getImgToOri().end(); it++) {
		if (*it==imageCombo->itemText(imageCombo->currentIndex())) continue;
		if (parametre->getImagesCorrel().contains(*it)) continue;	//l'image est déjà sélectionnée comme image pour la corrélation
		l.push_back(*it);
	}
	if (l.count()==0) return;

	ListeWindow* listeWindow = new ListeWindow(this,l);
	listeWindow->show();
	int rep = listeWindow->exec();
	if (rep!=QDialog::Accepted) return;

	QStringList l2 = listeWindow->getSelectedImages();
	delete listeWindow;
	parametre->modifImagesCorrel() << l2;
	correlImgsList->addItems(l2);
	if (correlImgsList->count()==paramMain->getParamApero().getImgToOri().count()-1) addCorrelImgButton->setEnabled(false);
	emit suiteMNT(true);
}

void MNTTab::addFromViewClicked() {
	//l'utilisateur doit sélectionner une caméra dans la vue 3D
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vue3D==0) vue3D = new VueChantier(paramMain, this, assistant);
	vue3D->show(SelectCamBox::CorrelImages, imageCombo->itemText(imageCombo->currentIndex()), parametre->getImagesCorrel());
	QApplication::restoreOverrideCursor();
	int rep = vue3D->exec();
	if (rep != QDialog::Accepted) return;
        if (vue3D->getRefImg().count()==0) return;
        QStringList cameras = vue3D->getRefImg();
        for (int i=0; i<cameras.count(); i++) {
                if (correlImgsList->findItems(cameras.at(i),Qt::MatchExactly).count()>0) continue;
		correlImgsList->addItem(cameras.at(i));
		parametre->modifImagesCorrel().push_back(cameras.at(i));
        }
	if (correlImgsList->count()==paramMain->getParamApero().getImgToOri().count()-1) addCorrelImgButton->setEnabled(false);
	emit suiteMNT(true);
}

void MNTTab::addFromStatClicked() {
	//ajoute par défaut les 4 caméras qui recouvrent le mieux la carte et encadrant la caméra de référence
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vue3D==0) vue3D = new VueChantier(paramMain, this, assistant);
       const  QVector<Pose>* poses = &(vue3D->getPoses());
		//pose de référence
	int n = 0;
	while (n<poses->count()) {
		if (poses->at(n).getNomImg()==imageCombo->itemText(imageCombo->currentIndex()))
			break;
		n++;
	}
	ElMatrix<REAL> rotation = poses->at(n).rotation();
	QVector<Pt3dr> emprise = poses->at(n).getEmprise();

		//recherche des 4 meilleures caméras
	QVector<pair<QString,double> > cam(8);
	QVector<REAL> V1 = poses->at(n).direction();	//normé
	for (int k=0; k<8; k++) cam[k].second = -1;
	for (int i=0; i<poses->count(); i++) {
		if (i==n) continue;

		//on ne prend que les caméras qui sont dans le même demi-espace que la caméra de référence (on calcule l'angle entre les directions)
		QVector<REAL> V2 = poses->at(i).direction();
		double angle = 0;	//cos angle(V1;V2)
		for (int k=0; k<3; k++) angle += V1[k] * V2[k]; //prod scal
		if (angle<0) continue;

		//recouvrement (projection de l'emprise au plan moyen de l'img de réf dans l'img i et % du recouvrement par/ à la projection)
		QVector<Pt2dr> proj(4);	//emprise de l'img de réf sur l'img i
		for (int j=0; j<4; j++)
			proj[j] = poses->at(i).getCamera().R3toF2(emprise[j]);
		REAL proj2[4] = {proj[0].x, proj[0].x, proj[0].y, proj[0].y};	//rectangle de l'emprise
		for (int j=1; j<4; j++) {
			if (proj[j].x<proj2[0]) proj2[0] = proj[j].x;
			else if (proj[j].x>proj2[1]) proj2[1] = proj[j].x;
			if (proj[j].y<proj2[2]) proj2[2] = proj[j].y;
			else if (proj[j].y>proj2[3]) proj2[3] = proj[j].y;
		}
		REAL rec[4] = { 0, (REAL)poses->at(i).width(),
						0, (REAL)poses->at(i).height() };	//rectangle de emprise n img i
		if (proj2[0]>rec[0]) rec[0] = proj2[0];
		else if (proj2[1]<rec[1]) rec[1] = proj2[1];
		if (proj2[2]>rec[2]) rec[2] = proj2[0];
		else if (proj2[3]<rec[3]) rec[3] = proj2[3];
		double recouvrement = (rec[1]-rec[0])*(rec[3]-rec[2]) / (proj2[1]-proj2[0])*(proj2[3]-proj2[2]);
		if (recouvrement<0.6) continue;

		//quadrant (placement du sommet i par rapport au sommet n dans le plan de l'img n)
		QVector<REAL> DS(3);
		for (int k=0; k<3; k++) DS[k] = poses->at(i).centre2()[k] - poses->at(n).centre2()[k];
		QVector<REAL> D(2,0);
		for (int k=0; k<2; k++)
			for (int l=0; l<2; l++) D[k] += rotation(l,k) * DS[l];

		//enregistrement (on veut la meilleure cam dans chaque direction (-x, +x, -y, +y), comme la cam i est dans 2 directions, on prend les 2 meilleures caméras dans chaque direction pour avoir au moins 4 caméras)
		for (int k=0; k<8; k++) {
			bool b = (D[0]<0 && (k==0 || k==1))
				|| (D[0]>0 && (k==2 || k==3))
				|| (D[1]<0 && (k==4 || k==5))
				|| (D[1]>0 && (k==6 || k==7));
			//if (b && recouvrement>cam[k].second) {
			if (b && angle>cam[k].second) {
				if (2*(k/2)==k) {
					cam[k+1].second = cam[k].second;
					cam[k+1].first = cam[k].first;
				}
				//cam[k].second = recouvrement;
				cam[k].second = angle;
				cam[k].first = poses->at(i).getNomImg();
				if (k<4) k=3;
				else break;
			}
		}	
	}

	//on vérifie qu'il y a une caméra dans chaque direction
	for (int k=0; k<8; k+=2) {
		if (cam[k].first.isEmpty()) {
			qMessageBox(this, tr("Warning"), QApplication::translate("Dialog", tr("Selected cameras do not surround the reference camera.\nSome parts of the depth map could be unseen from any other image and not be reconstructed.").toStdString().c_str(), 0, QApplication::CodecForTr));
			break;
		}
	}

	//on ne prend que 4 caméras
	QStringList l;
	for (int k=0; k<8; k+=2) {
		if (!l.contains(cam[k].first) && !cam[k].first.isEmpty()) l.push_back(cam[k].first);
	}
	int k = 1;
	while (k<8 && l.count()<4) {
		if (!l.contains(cam[k].first) && !cam[k].first.isEmpty()) l.push_back(cam[k].first);
		k+=2;
	}

	//affichage
	for (int k=0; k<l.count(); k++) {
                if (correlImgsList->findItems(l.at(k),Qt::MatchExactly).count()>0) continue;
		correlImgsList->addItem(l.at(k));	
		parametre->modifImagesCorrel().push_back(l.at(k));	
	}
	if (correlImgsList->count()==paramMain->getParamApero().getImgToOri().count()-1) addCorrelImgButton->setEnabled(false);
	QApplication::restoreOverrideCursor();
	emit suiteMNT(true);
}

void MNTTab::selectionChanged() { removeCorrelImgButton->setEnabled(correlImgsList->selectedItems().size()>0); }

void MNTTab::removeCorrelImgClicked() {
	if (correlImgsList->selectedItems().size()==0) return;
	QStringList l;
	for (int i=0; i<correlImgsList->count(); i++) {
		if (!correlImgsList->item(i)->isSelected())
			l.push_back(correlImgsList->item(i)->text());
	}
	correlImgsList->clear();
	correlImgsList->addItems(l);
	parametre->setImagesCorrel(l);
	if (correlImgsList->count()<paramMain->getParamApero().getImgToOri().count()-1) addCorrelImgButton->setEnabled(true);
	emit suiteMNT(correlImgsList->count()>0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


RepereTab::RepereTab(InterfMicmac* interfMicmac, const ParamMain* pMain, VueChantier* vueChantier,  CarteDeProfondeur* param, Assistant* help):
	QScrollArea(),
	resizableWidget( 0 ),
	assistant( help ),
	parametre( param ),
	parent( interfMicmac ),
	paramMain( pMain ),
	vue3D( vueChantier ),
	dir( pMain->getDossier() )
{
	//choix du repère
	QGroupBox *radioBox = NULL;
	if (paramMain->getParamPastis().getTypeChantier()==ParamPastis::Convergent) {
		radioTerrain = new QRadioButton(conv(tr("Euclidean frame")));
		radioTerrain->setToolTip(conv(tr("A depth map is a DSM computed in the Euclidean frame of pose orientation and bounded by the union of all images bounding boxes selected for correlation")));
		radioTerrain->setChecked(!parametre->getRepere());
		radioImage = new QRadioButton(conv(tr("Reference image frame")));
		radioImage->setToolTip(conv(tr("The depth map computation corresponds to the computation of the depth of each pixel of the reference image")));
		connect(radioTerrain, SIGNAL(clicked()), this, SLOT(repereClicked()));
		connect(radioImage, SIGNAL(clicked()), this, SLOT(repereClicked()));

		QHBoxLayout *radioLayout = new QHBoxLayout;
		radioLayout->addWidget(radioTerrain,0,Qt::AlignHCenter);
		radioLayout->addWidget(radioImage,0,Qt::AlignHCenter);
		radioLayout->addStretch(0);

		radioBox = new QGroupBox;
		radioBox->setLayout(radioLayout);
	} else
		parametre->setRepere(false);

	//nouveau repère terrain
	autreRepCheck = new QCheckBox(conv(tr("Other frame than pose estimation frame (horizontal plane)")));
	autreRepCheck->setToolTip(conv(tr("If checked an other frame must be defined.")));
	connect(autreRepCheck, SIGNAL(stateChanged(int)), this, SLOT(autreRepChecked()));

	//import du repère
	radioOpen = new QRadioButton(conv(tr("Load an existing frame")));
	radioOpen->setToolTip(conv(tr("Open an xml file if a frame has already been computed")));
	radioNew = new QRadioButton(conv(tr("Define a new frame")));
	radioNew->setToolTip(conv(tr("Draw graphically the new frame parameters (horizontal plane and axis)")));
	connect(radioOpen, SIGNAL(clicked()), this, SLOT(importRepClicked()));
	connect(radioNew, SIGNAL(clicked()), this, SLOT(importRepClicked()));

	QHBoxLayout *radio2Layout = new QHBoxLayout;
	radio2Layout->addWidget(radioOpen,0,Qt::AlignHCenter);	
	radio2Layout->addWidget(radioNew,0,Qt::AlignHCenter);
	radio2Layout->addStretch(0);

	radio2Box = new QGroupBox;
	radio2Box->setLayout(radio2Layout);

	//import d'un fichier
	QLabel* openLabel = new QLabel(tr("File :"));
	openEdit = new QLineEdit;
	openEdit->setMinimumWidth(150);
	openEdit->setEnabled(false);
	openButton = new QPushButton(tr("..."));
	openButton->setToolTip(tr("Open a file"));
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

	//paramétrage du repère (nouveau repère)
	vue3DButton = new QPushButton(QIcon(g_iconDirectory+"viewmag.png"), QString());
	vue3DButton->setToolTip(conv(tr("Select graphically background image in the 3D view")));
	vue3DButton->setMaximumSize (QSize(32,32));
	connect(vue3DButton, SIGNAL(clicked()), this, SLOT(vue3DClicked()));

	QLabel* planLabel = new QLabel(tr("Horizontal plane"));
	QFont font;
	font.setBold(true);
	planLabel->setFont(font);
	masqueWidget = new MasqueWidget(paramMain, assistant, true, false, vue3DButton, param->getImgRepMasq(), QString("_MasqRepTA"));
		mapper = new QSignalMapper(); 	
	connect(masqueWidget, SIGNAL(updateParam()), mapper, SLOT(map()));
	mapper->setMapping(masqueWidget, 0);
	
	QLabel* directionLabel = new QLabel(tr("Direction"));
	directionLabel->setFont(font);
	directionWidget = new DirectionWidget(paramMain, paramMain->getParamApero().getImgToOri(), assistant, pair<QString,QString>(param->getImgRep(),param->getImgRep()), 1, param->getSegmentRep(), param->getAxeRep());
	connect(directionWidget, SIGNAL(updateParam()), mapper, SLOT(map()));
	mapper->setMapping(directionWidget, 1);
		connect(mapper, SIGNAL(mapped(int)),this, SLOT(updateParam(int)));

	QVBoxLayout *paramLayout = new QVBoxLayout;
	paramLayout->addWidget(planLabel);
	paramLayout->addWidget(masqueWidget->getMasqueBox());
	paramLayout->addWidget(directionLabel);
	paramLayout->addWidget(directionWidget->getBox());
	paramLayout->addStretch();
	paramBox = new QGroupBox;
	paramBox->setFlat(true);
	paramBox->setLayout(paramLayout);

	//bouton pour lancer le calcul du TA
	TAButton = new QPushButton(tr("Compute IM"));
	TAButton->setToolTip(conv(tr("Compute index map to draw mask in Euclidian frame")));
	TAButton->adjustSize();
	connect(TAButton, SIGNAL(clicked()), this, SLOT(TAClicked()));

	QVBoxLayout *mainLayout = new QVBoxLayout;
	if (paramMain->getParamPastis().getTypeChantier()==ParamPastis::Convergent) mainLayout->addWidget(radioBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(autreRepCheck,0,Qt::AlignHCenter);
	mainLayout->addWidget(radio2Box,0,Qt::AlignHCenter);
	mainLayout->addWidget(openBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(paramBox,0,Qt::AlignHCenter);
	mainLayout->addWidget(TAButton,0,Qt::AlignHCenter);
	mainLayout->addStretch();

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
	updateInterface(Begin);
}
RepereTab::~RepereTab() {
	delete mapper;
	delete masqueWidget;
	delete directionWidget;
}

QString RepereTab::calcTAName() { return paramMain->getDossier()+QString("TA%1/TA_Geom-Im-%1.tif").arg(paramMain->getNumImage(parametre->getImageDeReference())); }

void RepereTab::updateParam(int idx) {
	switch (idx) {
		case 0 : masqueWidget->updateParam(parametre,true);
			break;
		case 1 : directionWidget->updateParam(parametre);
	}
}

void RepereTab::updateInterface(Mode mode) {
	switch (mode) {
		case Begin : 
				if (paramMain->getParamPastis().getTypeChantier()==ParamPastis::Convergent) {
					radioTerrain->setChecked(false);
					radioImage->setChecked(false);
				}
				autreRepCheck->hide();
				autreRepCheck->setChecked(false);
				radio2Box->hide();
				radioOpen->setChecked(false);
				radioNew->setChecked(false);
				openBox->hide();
				openEdit->setText(QString());
				paramBox->hide();
				TAButton->hide();
				if (!parametre->getRepere()) updateInterface(Terrain);
				else emit suiteRepere(false);
				break;
		case Image :
				updateInterface(Begin);
				if (paramMain->getParamPastis().getTypeChantier()==ParamPastis::Convergent) radioImage->setChecked(true);	
				emit suiteRepere(true);	
				break;
		case Terrain :
				if (paramMain->getParamPastis().getTypeChantier()==ParamPastis::Convergent) {
					radioImage->setChecked(false);	
					radioTerrain->setChecked(true);
				}
				autreRepCheck->show();
				TAButton->show();		
				if (parametre->getAutreRepere()) updateInterface(AutreRepere);
				else {
					autreRepCheck->setChecked(false);
					radio2Box->hide();
					radioOpen->setChecked(false);
					radioNew->setChecked(false);
					openBox->hide();
					openEdit->setText(QString());
					paramBox->hide();
					emit suiteRepere(true);	
				}
				break;
		case AutreRepere :
				autreRepCheck->setChecked(true);
				radio2Box->show();	
				TAButton->hide();
				if (QFile(dir+parametre->getRepereFile(*paramMain)).exists()) updateInterface(OpenRepere);
				else {
					radioOpen->setChecked(false);
					radioNew->setChecked(false);
					openBox->hide();
					openEdit->setText(QString());
					paramBox->hide();
					emit suiteRepere(false);	
				}
				break;
		case NewRepere :
				radioOpen->setChecked(false);
				radioNew->setChecked(true);
				openBox->hide();
				openEdit->setText(QString());
				paramBox->show();
				TAButton->show();	//à modifier-> show quand masqueWidget et DirectionWidget ok
				emit suiteRepere(false);
				break;
		case OpenRepere :
				radioOpen->setChecked(true);
				radioNew->setChecked(false);
				openBox->show();
				if (QFile(dir+parametre->getRepereFile(*paramMain)).exists() && FichierRepere::lire(dir+parametre->getRepereFile(*paramMain)).isEmpty()) openEdit->setText(dir+parametre->getRepereFile(*paramMain));
				paramBox->hide();
				TAButton->show();
				emit suiteRepere(true);	
				break;
		case TACalcule :
				emit suiteRepere(true);	//recharge l'onglet masque			
				break;
	}
	adjustSize();
}

void RepereTab::repereClicked() {
	if (paramMain->getParamPastis().getTypeChantier()!=ParamPastis::Convergent) return;
	if (radioImage->isChecked()) {
		parametre->setRepere(true);
		parametre->setAutreRepere(false);
		updateInterface(Image);
	} else {
		parametre->setRepere(false);
		updateInterface(Terrain);
	}
}

void RepereTab::autreRepChecked() {
	if (autreRepCheck->isChecked()) {
		parametre->setAutreRepere(true);
		updateInterface(AutreRepere);
	} else {
		parametre->setAutreRepere(false);
		updateInterface(Terrain);
	}
}

void RepereTab::importRepClicked() {
	if (radioOpen->isChecked()) {
		updateInterface(OpenRepere);
	} else {
		updateInterface(NewRepere);
	}
}

void RepereTab::openClicked() {
	FileDialog fileDialog(this, tr("Open a frame file"), dir, tr("Frame files (*.xml)") );
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
	QStringList fileNames;
	if (fileDialog.exec()) fileNames = fileDialog.selectedFiles();
	else return;
  	if (fileNames.size()==0) return;
  	if (*(fileNames.begin())==openEdit->text()) return;
	QString fichier = *(fileNames.begin());
	fichier = QDir(dir).absoluteFilePath(fichier);	//chemin absolu
	if (!checkPath(fichier)) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read file %1.\nCheck there are no accents in path.")).arg(fichier));	
		return;
	}
	QString err = FichierRepere::lire(fichier);
	if (!err.isEmpty()) {
		qMessageBox(this, tr("Read error"),err);	
		return;
	}
	QFile(fichier).rename(dir+parametre->getRepereFile(*paramMain));
	openEdit->setText(dir+parametre->getRepereFile(*paramMain));
}

void RepereTab::vue3DClicked() {	
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (vue3D==0) vue3D = new VueChantier(paramMain, this, assistant);
	vue3D->show(SelectCamBox::RefImage);
	QApplication::restoreOverrideCursor();
	int rep = vue3D->exec();
	if (rep != QDialog::Accepted) return;
        if (vue3D->getRefImg().count()==0) return;
        QString fichier = vue3D->getRefImg().at(0);
	masqueWidget->setImageFond(fichier);
}

QString RepereTab::QPoint2QString(const QPoint& P) { return QString("[%1,%2]").arg(P.x()).arg(P.y()); }

void RepereTab::TAClicked() {
//calcul du repère
	QString s = noBlank(dir) + QString("\\(");
	for (QStringList::const_iterator it=paramMain->getParamApero().getImgToOri().begin(); it!=paramMain->getParamApero().getImgToOri().end(); it++) {
		s += *it;
		if (paramMain->getParamApero().getImgToOri().end()-it!=1) s += QString("\\|");
	}
	s += QString("\\)");

	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (parametre->getAutreRepere() && radioNew->isChecked()) {
		//vérification des paramètres
		if (parametre->getImgRepMasq().isEmpty() || !QFile(parametre->getImgRepMasq().section(".",0,-2)+QString("_MasqRepTA.tif")).exists() || parametre->getImgRep().isEmpty() || parametre->getSegmentRep().first==parametre->getSegmentRep().second) {
			if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
			qMessageBox(this, conv(tr("Parameter error")),  conv(tr("Index map parameters are unvalid.")));
			return;
		}

		//formats
		/*for (QList<CalibCam>::const_iterator it=paramMain->getParamPastis().getCalibs().begin(); it!=paramMain->getParamPastis().getCalibs().end(); it++) {
			QString sf = QVariant(it->getFocale()).toString();
			while (sf.count()!=3) sf = QString("0")+sf;
			if (!QFile(dir+QString("Ori-F/AutoCal%1").arg(it->getFocale())+QString("0.xml")).exists()) QFile(dir+QString("Ori-F/F%1_AutoCalFinale.xml").arg(sf)).copy(dir+QString("Ori-F/AutoCal%1").arg(it->getFocale())+QString("0.xml"));
		}*/
		if (!FichierImgToOri::ecrire (dir+QString("KeyCalibration.xml"), QStringList(parametre->getImagesCorrel())<<parametre->getImageDeReference(), paramMain->getCorrespImgCalib(), QString(), paramMain->getParamPastis().getCalibFiles(), false, true, 2, QList<int>())) {
			if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
			qMessageBox(this, conv(tr("Write error")),  conv(tr("Fail to write images to be reoriented.")));
			return;
		}
cout << (dir+QString("KeyCalibration.xml")).toStdString() << endl;		

		//calcul
		QString commande = comm(QString("cd %1 \n %1bin/Bascule %2 F %3 PostPlan=%4 P1Rep=%5 P2Rep=%6 AxeRep=%7 ImRep=%8").arg(noBlank(paramMain->getMicmacDir())).arg(s).arg(noBlank(parametre->getRepereFile(*paramMain))).arg("_MasqRepTA").arg(QPoint2QString(parametre->getSegmentRep().first)).arg(QPoint2QString(parametre->getSegmentRep().second)).arg(QPoint2QString(parametre->getAxeRep())).arg(parametre->getImgRep()));
		if (execute(commande)!=0) {
			if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
			qMessageBox(this, conv(tr("Execution error")),  conv(tr("Fail to write images to be reoriented.")));
			return;
		}
		deleteFile(dir+QString("KeyCalibration.xml"));
	} else if (!parametre->getAutreRepere()) {	//repère par défaut
		QString virgule;
		systemeNumerique(virgule);
		QString file = paramMain->getDossier() + QString("Ori-F/") + QString("OrFinale-") + parametre->getImageDeReference().section(".",0,-2) + QString(".xml");
		QString orient = file.section(".",0,-2) + QString("2.xml");
		QString fichier = (virgule==QString("."))? file : orient;
		CamStenope* cam = NS_ParamChantierPhotogram::Cam_Gen_From_File(fichier.toStdString(), string("OrientationConique"), 0)->CS();	//ElCamera::CS = static_cast<CamStenope*>(ElCamera) (pas de delete ElCamera)
		if (cam==0) {
			if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
			qMessageBox(this, conv(tr("Read error")), conv(tr("Fail to read reference image pose.")));
			return;
		}		
		double profondeur = cam->GetProfondeur();

		if (!FichierRepere::ecrire(dir+parametre->getRepereFile(*paramMain), -1*profondeur)) {
			if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
			qMessageBox(this, conv(tr("Execution error")), conv(tr("Fail to write index map default frame.")));
			return;
		}
	}

//calcul du TA
	QString s2 = noBlank(dir) + QString("\\(");
	for (QStringList::const_iterator it=parametre->getImagesCorrel().begin(); it!=parametre->getImagesCorrel().end(); it++) {
		s2 += *it + QString("\\|");
	}
	s2 += QString("%1\\)").arg(parametre->getImageDeReference());
	rm(dir+QString("TA"));
	rm(dir+QString("Pyram"));
	rm(dir+QString("TA%1").arg(paramMain->getNumImage(parametre->getImageDeReference())));
	rm(dir+QString("Pyram%1").arg(paramMain->getNumImage(parametre->getImageDeReference())));
	QString commande = comm(QString("cd %1 \n %1bin/Tarama %2 F Repere=%3 Zoom=8").arg(noBlank(paramMain->getMicmacDir())).arg(s2).arg(noBlank(parametre->getRepereFile(*paramMain))));
	if (execute(commande)!=0) {
		if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
		qMessageBox(this, conv(tr("Execution error")),  conv(tr("Index map computation failed.")));
		return;
	}
	QDir(dir).rename(QString("TA"),QString("TA%1").arg(paramMain->getNumImage(parametre->getImageDeReference())));
	QDir(dir).rename(QString("Pyram"),QString("Pyram%1").arg(paramMain->getNumImage(parametre->getImageDeReference())));

	//TA couleur
	QString err = MasqueWidget::convert2Rgba(parametre->getImageSaisie(*paramMain), false);
	if (!err.isEmpty()) {
		if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
		qMessageBox(this, conv(tr("Execution error")),err);
		return;
	}

	//écriture du fichier de référencement du masque
	ParamMasqueXml paramMasqueXml;
	QString infile = paramMain->getDossier()+QString("TA%1/Z_Num1_DeZoom8_LeChantier.xml").arg(paramMain->getNumImage(parametre->getImageDeReference()));
	err = FichierMasque::lire(infile, paramMasqueXml);
	if (!err.isEmpty()) {
		if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
		qMessageBox(this, conv(tr("Read error")),err);
		return;
	}
	paramMasqueXml.setNameFileMnt(parametre->getReferencementMasque(*paramMain));
	paramMasqueXml.setNameFileMasque(parametre->getMasque(*paramMain));
	if (!FichierMasque::ecrire(parametre->getReferencementMasque(*paramMain), paramMasqueXml)) {
		if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
		qMessageBox(this, conv(tr("Write error")), conv(tr("Fail to create mask referencing file.")));
		return;
	}

	if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
	qMessageBox(this, conv(tr("Information")),  conv(tr("Index map computed.")));
	updateInterface(TACalcule);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MaskTab::MaskTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  CarteDeProfondeur* param, Assistant* help):
	QScrollArea(),
	assistant( help ),
	parametre( param ),
	parent( interfMicmac ),
	paramMain( pMain ),
	dir( pMain->getDossier() )
{
	masqueWidget = new MasqueWidget(paramMain, assistant, true, true, 0, parametre->getImageSaisie(*paramMain), QString("_masque"));
	connect(masqueWidget, SIGNAL(updateParam()), this, SLOT(updateParam()));
	connect(masqueWidget, SIGNAL(updateParam()), this, SIGNAL(suiteMasque()));

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(masqueWidget->getMasqueBox());

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
}
MaskTab::~MaskTab() {
	delete masqueWidget;
}

void MaskTab::updateParam() { masqueWidget->updateParam(parametre,false); }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


OrthoTab::OrthoTab(InterfMicmac* interfMicmac, const ParamMain* pMain, CarteDeProfondeur* param, Assistant* help):
	QScrollArea(),
	assistant( help ),
	parent( interfMicmac ),
	paramMain( pMain ),
	parametre( param ),
	dir( pMain->getDossier() )
{
	//orthoimage ?
	checkOrtho = new QCheckBox(conv(tr("Compute orthoimages")));
		QFont font;
		font.setBold(true);
	checkOrtho->setFont(font);
	checkOrtho->setChecked(parametre->getDoOrtho());
	connect(checkOrtho, SIGNAL(stateChanged(int)), this, SLOT(orthoClicked()));

	//images
	QLabel* imgLabel = new QLabel(conv(tr("Images to orthorectify :")));
	listeWidget = new QListWidget;
	listeWidget->setSelectionMode (QAbstractItemView::ExtendedSelection);
	if (parametre->getImgsOrtho().count()>0) listeWidget->addItems(parametre->getImgsOrtho());
	else {
		listeWidget->addItems(parametre->getImagesCorrel());
		listeWidget->addItem(parametre->getImageDeReference());
		parametre->setImgsOrtho(parametre->getImagesCorrel());
		parametre->modifImgsOrtho().push_back(parametre->getImageDeReference());
	}
	connect(listeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
	
	addImgsButton = new QPushButton;
	addImgsButton->setIcon(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"));
	addImgsButton->setMaximumSize (QSize(32,32));
	addImgsButton->setToolTip(conv(tr("Add an image to orthorectify")));
	addImgsButton->setEnabled(parametre->getImgsOrtho().count()<paramMain->getParamApero().getImgToOri().count());
	connect(addImgsButton, SIGNAL(clicked()), this, SLOT(addImgsClicked()));
	
	removeImgsButton = new QPushButton;
	removeImgsButton->setIcon(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic2.png"));
	removeImgsButton->setMaximumSize (QSize(32,32));
	removeImgsButton->setToolTip(conv(tr("Remove an image from orthorectification")));
	removeImgsButton->setEnabled(parametre->getImgsOrtho().count()<paramMain->getParamApero().getImgToOri().count());
	connect(removeImgsButton, SIGNAL(clicked()), this, SLOT(removeImgsClicked()));

	QHBoxLayout *imgsLayout = new QHBoxLayout;
	imgsLayout->addWidget(imgLabel);
	imgsLayout->addWidget(listeWidget);
	imgsLayout->addWidget(addImgsButton);
	imgsLayout->addWidget(removeImgsButton);
	imgsLayout->addStretch();

	imgsBox = new QGroupBox;
	imgsBox->setFlat(false);
	imgsBox->setLayout(imgsLayout);

	//échelle relative
	QLabel* echelleLabel = new QLabel(conv(tr("Relative scale :")));
	echelleLabel->setFont(font);
	echelleLabel->setToolTip(conv(tr("Relative scale between DTM and orthoimages.\nIt can be computed from two distances drawn on the images (upper part) or directly entered in the box (bottom part).")));

	QStringList l(parametre->getImagesCorrel());
	l<<parametre->getImageDeReference();
	QVector<QString> V(4);
	V[0] = parametre->getImgEchOrtho().first;
	V[1] = parametre->getImgEchOrtho().first;
	V[2] = parametre->getImgEchOrtho().second;
	V[3] = parametre->getImgEchOrtho().second;
	echelleWidget = new EchelleWidget(paramMain,
					2, parametre->getImgsOrtho(), l,
					assistant,
					pair<QVector<QString>,QVector<QPoint> >(V,parametre->getPtsEchOrtho()), parametre->getEchelleOrtho());
	connect(echelleWidget, SIGNAL(updateParam()), this, SLOT(updateParam()));
	connect(echelleWidget, SIGNAL(updateParam()), this, SIGNAL(suiteOrtho()));

	QHBoxLayout *echelleLayout = new QHBoxLayout;
	echelleLayout->addWidget(echelleLabel);
	echelleLayout->addWidget(echelleWidget->getBox());
	echelleLayout->addStretch();

	echelleBox = new QGroupBox;
	echelleBox->setFlat(false);
	echelleBox->setLayout(echelleLayout);

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(checkOrtho);
	mainLayout->addWidget(imgsBox);
	mainLayout->addWidget(echelleBox);
	mainLayout->addStretch();
	if (!parametre->getDoOrtho()) {
		imgsBox->hide();
		echelleBox->hide();
	}

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
	if (listeWidget->count()==paramMain->getParamApero().getImgToOri().count()) addImgsButton->setEnabled(false);
}
OrthoTab::~OrthoTab() {
	delete echelleWidget;
}

void OrthoTab::orthoClicked() {
	if (checkOrtho->isChecked()) {
		parametre->setDoOrtho(true);
		imgsBox->show();
		echelleBox->show();
	} else {
		parametre->setDoOrtho(false);
		imgsBox->hide();
		echelleBox->hide();
	}
	emit suiteOrtho();
}

void OrthoTab::selectionChanged() { removeImgsButton->setEnabled(listeWidget->selectedItems().size()>0); }

void OrthoTab::addImgsClicked() {
	//l'utilisateur doit sélectionner une caméra dans une liste
		//liste des images sélectionnables
	QStringList l;
	for (QStringList::const_iterator it=paramMain->getParamApero().getImgToOri().begin(); it!=paramMain->getParamApero().getImgToOri().end(); it++) {
		if (parametre->getImgsOrtho().contains(*it)) continue;	//l'image est déjà sélectionnée comme image pour la corrélation
		l.push_back(*it);
	}
	if (l.count()==0) return;

	ListeWindow* listeWindow = new ListeWindow(this,l);
	listeWindow->show();
	int rep = listeWindow->exec();
	if (rep!=QDialog::Accepted) return;

	QStringList l2 = listeWindow->getSelectedImages();
	parametre->modifImgsOrtho() << l2;
	listeWidget->addItems(l2);
	if (listeWidget->count()==paramMain->getParamApero().getImgToOri().count()) addImgsButton->setEnabled(false);
	delete listeWindow;
	emit suiteOrtho();
	QStringList l3;
	for (int i=0; i<listeWidget->count(); i++)
		l3 << listeWidget->item(i)->text();
	echelleWidget->updateListe2(l3);
}

void OrthoTab::removeImgsClicked() {
	if (listeWidget->selectedItems().size()==0) return;
	QStringList l;
	for (int i=0; i<listeWidget->count(); i++) {
		if (!listeWidget->item(i)->isSelected())
			l.push_back(listeWidget->item(i)->text());
	}
	listeWidget->clear();
	listeWidget->addItems(l);
	parametre->setImgsOrtho(l);
	removeImgsButton->setEnabled(false);
	emit suiteOrtho();
	QStringList l3;
	for (int i=0; i<listeWidget->count(); i++)
		l3 << listeWidget->item(i)->text();
	echelleWidget->updateListe2(l3);
}

void OrthoTab::updateParam() { echelleWidget->updateParam(parametre); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ProfondeurTab::ProfondeurTab(InterfMicmac* interfMicmac, const ParamMain* pMain, CarteDeProfondeur* param) : QScrollArea(), parent(interfMicmac), paramMain(pMain), parametre(param)
{
	//intervalle de profondeur
	QLabel* intervLabel = new QLabel(conv(tr("Depth interval for matching :")));
	QFont font;
	font.setBold(true);
	intervLabel->setFont(font);
	QLabel* intervMinLabel = new QLabel(conv(tr("Minimum depth :")));
	intervMinEdit = new QLineEdit;
	intervMinEdit->setMaximumWidth (100);
	intervMinEdit->setText(QVariant(parametre->getInterv().first).toString());
	intervMinEdit->setToolTip (conv(tr("Minimum depth at which correlation can be computed ; it is relative to previosly computed 3D tie-point mean depth.")));
	connect(intervMinEdit, SIGNAL(textChanged(QString)), this, SLOT(updateParam()));
	QLabel* intervMaxLabel = new QLabel(conv(tr("Maximum depth :")));
	intervMaxEdit = new QLineEdit;
	intervMaxEdit->setMaximumWidth (100);
	intervMaxEdit->setText(QVariant(parametre->getInterv().second).toString());
	intervMaxEdit->setToolTip (conv(tr("Maximum depth at which correlation can be computed ; it is relative to previosly computed 3D tie-point mean depth.")));
	connect(intervMaxEdit, SIGNAL(textChanged(QString)), this, SLOT(updateParam()));

	//filtrage des discontinuités	
	checkDiscont = new QCheckBox(conv(tr("Take discontinuities and strong slopes into account")));
	checkDiscont->setFont(font);
	checkDiscont->setChecked(parametre->getDiscontinuites());
	checkDiscont->setToolTip (conv(tr("If it is selected, areas of high depth gradient will not be smoothed to take discontinuities and strong slopes into account.")));
	connect(checkDiscont, SIGNAL(stateChanged(int)), this, SLOT(discontClicked()));

	QLabel* regulAbsLabel = new QLabel(conv(tr("Depth gradient threshold :")));
	regulAbsEdit = new QLineEdit;
	regulAbsEdit->setMaximumWidth (100);
	regulAbsEdit->setText(QVariant(parametre->getSeuilZ()).toString());
	regulAbsEdit->setToolTip (conv(tr("If it is selected, areas of high depth gradient will not be smoothed to take discontinuities and strong slopes into account.")));
	connect(regulAbsEdit, SIGNAL(textChanged(QString)), this, SLOT(updateParam()));
	QLabel* regulLabel = new QLabel(conv(tr("Smoothing weight coefficient for hight depth gradient :")));	
	regulEdit = new QLineEdit;
	regulEdit->setMaximumWidth (100);
	regulEdit->setText(QVariant(parametre->getSeuilZRelatif()).toString());
	regulEdit->setToolTip (conv(tr("If it is selected, areas of high depth gradient will not be smoothed to take discontinuities and strong slopes into account.")));
	connect(regulEdit, SIGNAL(textChanged(QString)), this, SLOT(updateParam()));

	QFormLayout *discontLayout = new QFormLayout;
	discontLayout->addRow(regulAbsLabel,regulAbsEdit);
	discontLayout->addRow(regulLabel,regulEdit);
	discontLayout->setFormAlignment(Qt::AlignCenter);

	discontBox = new QGroupBox;
	discontBox->setLayout(discontLayout);

	QFormLayout *paramLayout = new QFormLayout;
	paramLayout->addRow(intervLabel);
	paramLayout->addRow(intervMinLabel,intervMinEdit);
	paramLayout->addRow(intervMaxLabel,intervMaxEdit);
	paramLayout->addRow(checkDiscont);
	paramLayout->addRow(discontBox);

	QGroupBox* paramBox = new QGroupBox;
	paramBox->setLayout(paramLayout);

	QFormLayout* mainLayout = new QFormLayout;
	mainLayout->addWidget(paramBox);
	mainLayout->setFormAlignment(Qt::AlignCenter);

	resizableWidget = new QWidget;
	resizableWidget->setLayout(mainLayout);
	setWidget(resizableWidget);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	adjustSize();
	resizableWidget->layout()->setSizeConstraint(QLayout::SetFixedSize);
	discontClicked();
}
ProfondeurTab::~ProfondeurTab() {}

void ProfondeurTab::discontClicked() {
	if (checkDiscont->isChecked()) discontBox->show();
	else discontBox->hide();
	parametre->setDiscontinuites(checkDiscont->isChecked());
}

void ProfondeurTab::updateParam() {
	//intervalle
	bool ok;
	double pmin = intervMinEdit->text().toDouble(&ok);
	if (!ok) return;

	float pmax = intervMaxEdit->text().toDouble(&ok);
	if (!ok) return;
	parametre->setInterv( pair<float,float>((float)pmin,pmax) );

	//discontinuités
	if (checkDiscont->isChecked()) {
		float regabs = regulAbsEdit->text().toDouble(&ok);
		if (!ok) return;
		parametre->setSeuilZ(regabs);

		float regrel = regulEdit->text().toDouble(&ok);
		if (!ok) return;
		parametre->setSeuilZRelatif(regrel);
	}
	emit suiteProf();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ListeWindow::ListeWindow(QWidget* parent, const QStringList& images) : QDialog(parent) {	
	liste = new QListWidget;
	liste->addItems(images);
	liste->setSelectionMode (QAbstractItemView::ExtendedSelection);

	QDialogButtonBox* buttonBox = new QDialogButtonBox();

	QPushButton* cancelButton = buttonBox->addButton (tr("Cancel"), QDialogButtonBox::ActionRole);
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));

	QPushButton* okButton = buttonBox->addButton (tr("Ok"), QDialogButtonBox::AcceptRole);
	okButton->setToolTip(tr("Launch computation of all depth maps"));
	connect(okButton, SIGNAL(clicked()), this, SLOT(accept()));

	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(liste, 0, Qt::AlignLeft);
	mainLayout->addWidget(buttonBox, 0, Qt::AlignRight);
	mainLayout->addStretch();
	setLayout(mainLayout);
	layout()->setSizeConstraint(QLayout::SetFixedSize);
}
ListeWindow::~ListeWindow() {}

QStringList ListeWindow::getSelectedImages() {
	QStringList l;
	QList<QListWidgetItem*> l2 = liste->selectedItems();
	for (QList<QListWidgetItem*>::const_iterator it=l2.begin(); it!=l2.end(); it++)
		l.push_back((*it)->text());
	return l;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


CarteDeProfondeur::CarteDeProfondeur():
	aCalculer(true),
	imageDeReference(QString()),
	imagesCorrel(QStringList()),
	repere(true),
	autreRepere(false),
	imgRepMasq(QString()),
	imgRep(QString()),
	segmentRep(std::pair<QPoint,QPoint>(QPoint(-1,-1),QPoint(-1,-1))),
	axeRep(QPoint(1,0)),
	doOrtho(false),
	orthoCalculee(false),
	imgsOrtho(QStringList()),
	echelleOrtho(1.0),
	imgEchOrtho(std::pair<QString,QString>()),
	ptsEchOrtho(QVector<QPoint>()),
	interv(std::pair<float,float>(0.3f,5.f)),
	discontinuites(false),
	seuilZ(0),
	seuilZRelatif(0.3f) {}
CarteDeProfondeur::CarteDeProfondeur(const CarteDeProfondeur& carteDeProfondeur) { copie(carteDeProfondeur); }
CarteDeProfondeur::~CarteDeProfondeur() {}

CarteDeProfondeur& CarteDeProfondeur::operator=(const CarteDeProfondeur& carteDeProfondeur) {
	if (this!=&carteDeProfondeur) copie(carteDeProfondeur);
	return *this;
}

void CarteDeProfondeur::copie(const CarteDeProfondeur& carteDeProfondeur) {
	aCalculer = carteDeProfondeur.getACalculer();
	imageDeReference = carteDeProfondeur.getImageDeReference();
	imagesCorrel = carteDeProfondeur.getImagesCorrel();
	repere = carteDeProfondeur.getRepere();
	autreRepere = carteDeProfondeur.getAutreRepere();
	imgRepMasq = carteDeProfondeur.getImgRepMasq();
	imgRep = carteDeProfondeur.getImgRep();
	segmentRep = carteDeProfondeur.getSegmentRep();
	axeRep = carteDeProfondeur.getAxeRep();
	doOrtho = carteDeProfondeur.getDoOrtho();
	orthoCalculee = carteDeProfondeur.getOrthoCalculee();
	imgsOrtho = carteDeProfondeur.getImgsOrtho();
	echelleOrtho = carteDeProfondeur.getEchelleOrtho();
	imgEchOrtho = carteDeProfondeur.getImgEchOrtho();
	ptsEchOrtho = carteDeProfondeur.getPtsEchOrtho();
	interv = carteDeProfondeur.getInterv();
	discontinuites = carteDeProfondeur.getDiscontinuites();
	seuilZ = carteDeProfondeur.getSeuilZ();
	seuilZRelatif = carteDeProfondeur.getSeuilZRelatif();
}

bool CarteDeProfondeur::getACalculer() const { return aCalculer; }
const QString& CarteDeProfondeur::getImageDeReference() const { return imageDeReference; }
const QStringList& CarteDeProfondeur::getImagesCorrel() const { return imagesCorrel; }
QStringList& CarteDeProfondeur::modifImagesCorrel() { return imagesCorrel; }
bool CarteDeProfondeur::getRepere() const { return repere; }
bool CarteDeProfondeur::getAutreRepere() const { return autreRepere; }
const QString& CarteDeProfondeur::getImgRepMasq() const { return imgRepMasq; }
const QString& CarteDeProfondeur::getImgRep() const { return imgRep; }
const std::pair<QPoint,QPoint>& CarteDeProfondeur::getSegmentRep() const { return segmentRep; }
const QPoint& CarteDeProfondeur::getAxeRep() const { return axeRep; }
bool CarteDeProfondeur::getDoOrtho() const { return doOrtho; }
bool CarteDeProfondeur::getOrthoCalculee() const { return orthoCalculee; }
const QStringList& CarteDeProfondeur::getImgsOrtho() const { return imgsOrtho; }
QStringList& CarteDeProfondeur::modifImgsOrtho() { return imgsOrtho; }
double CarteDeProfondeur::getEchelleOrtho() const { return echelleOrtho; }
const std::pair<QString,QString>& CarteDeProfondeur::getImgEchOrtho() const { return imgEchOrtho; }
const QVector<QPoint>& CarteDeProfondeur::getPtsEchOrtho() const { return ptsEchOrtho; }
const std::pair<float,float>& CarteDeProfondeur::getInterv() const { return interv; }
bool CarteDeProfondeur::getDiscontinuites() const { return discontinuites; }
float CarteDeProfondeur::getSeuilZ() const { return seuilZ; }
float CarteDeProfondeur::getSeuilZRelatif() const { return seuilZRelatif; }

void CarteDeProfondeur::setACalculer(bool a) { aCalculer = a; }
void CarteDeProfondeur::setImageDeReference(const QString& i) { imageDeReference = i; }
void CarteDeProfondeur::setImagesCorrel(const QStringList& im) { imagesCorrel = im; }
void CarteDeProfondeur::setRepere(bool r) { repere = r; }
void CarteDeProfondeur::setAutreRepere(bool n) { autreRepere = n; }
void CarteDeProfondeur::setImgRepMasq(const QString& img) { imgRepMasq = img; }
void CarteDeProfondeur::setImgRep(const QString& imgR) { imgRep = imgR; }
void CarteDeProfondeur::setSegmentRep(const std::pair<QPoint,QPoint>& s) { segmentRep = s; }
void CarteDeProfondeur::setAxeRep(const QPoint& a) { axeRep = a; }
void CarteDeProfondeur::setDoOrtho(bool d) { doOrtho = d; }
void CarteDeProfondeur::setOrthoCalculee(bool o) { orthoCalculee = o; }
void CarteDeProfondeur::setImgsOrtho(const QStringList& imgs) { imgsOrtho = imgs; }
void CarteDeProfondeur::setEchelleOrtho(double e) { echelleOrtho = e; }
void CarteDeProfondeur::setImgEchOrtho(const std::pair<QString,QString>& imgE) { imgEchOrtho = imgE; }
void CarteDeProfondeur::setPtsEchOrtho(const QVector<QPoint>& p) { ptsEchOrtho = p; }
void CarteDeProfondeur::setInterv(const std::pair<float,float>& in) { interv = in; }
void CarteDeProfondeur::setDiscontinuites(bool di) { discontinuites = di; }
void CarteDeProfondeur::setSeuilZ(float se) { seuilZ = se; }
void CarteDeProfondeur::setSeuilZRelatif(float seu) { seuilZRelatif = seu; }

QString CarteDeProfondeur::getImageSaisie(const ParamMain& paramMain) const {
	if (repere) return paramMain.getDossier()+imageDeReference;
	else return paramMain.getDossier()+QString("TA%1/TA_LeChantier.tif").arg(paramMain.getNumImage(imageDeReference));
}
QString CarteDeProfondeur::getRepereFile(const ParamMain& paramMain) const { return QString("Repere_%1.xml").arg(paramMain.getNumImage(imageDeReference)); }
QString CarteDeProfondeur::getMasque(const ParamMain& paramMain) const { return paramMain.getDossier()+QString("Masque_%1.tif").arg(paramMain.getNumImage(imageDeReference)); }
QString CarteDeProfondeur::getReferencementMasque(const ParamMain& paramMain) const { return getMasque(paramMain).section(".",0,-2)+QString(".xml"); }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ParamMasqueXml::ParamMasqueXml() : NameFileMnt(QString()), NameFileMasque(QString()), NombrePixels(QSize(0,0)), OriginePlani(QPointF(0,0)), ResolutionPlani(QPointF(1,1)), OrigineAlti(0), ResolutionAlti(1), Geometrie(QString("eGeomMNTFaisceauIm1PrCh_Px1D"))
{}
ParamMasqueXml::ParamMasqueXml(const QString& mntFile, const QString& masqueFile, const QSize& nbPx, const QPointF& OrigXY, const QPointF& ResolXY, double OrigZ, double ResolZ, const QString& Geom) : NameFileMnt(mntFile), NameFileMasque(masqueFile), NombrePixels(nbPx), OriginePlani(OrigXY), ResolutionPlani(ResolXY), OrigineAlti(OrigZ), ResolutionAlti(ResolZ), Geometrie(Geom)
{}
ParamMasqueXml::ParamMasqueXml(const ParamMasqueXml& paramMasqueXml) { copie(paramMasqueXml); }
ParamMasqueXml::~ParamMasqueXml() {}

ParamMasqueXml& ParamMasqueXml::operator=(const ParamMasqueXml& paramMasqueXml) {
	if (this!=&paramMasqueXml) copie(paramMasqueXml);
	return *this;
}

void ParamMasqueXml::copie(const ParamMasqueXml& paramMasqueXml) {
	NameFileMnt = paramMasqueXml.getNameFileMnt();
	NameFileMasque = paramMasqueXml.getNameFileMasque();
	NombrePixels = paramMasqueXml.getNombrePixels();
	OriginePlani = paramMasqueXml.getOriginePlani();
	ResolutionPlani = paramMasqueXml.getResolutionPlani();
	OrigineAlti = paramMasqueXml.getOrigineAlti();
	ResolutionAlti = paramMasqueXml.getResolutionAlti();
	Geometrie = paramMasqueXml.getGeometrie();
}

const QString& ParamMasqueXml::getNameFileMnt() const {return NameFileMnt;}
const QString& ParamMasqueXml::getNameFileMasque() const {return NameFileMasque;}
const QSize& ParamMasqueXml::getNombrePixels() const {return NombrePixels;}
const QPointF& ParamMasqueXml::getOriginePlani() const {return OriginePlani;}
const QPointF& ParamMasqueXml::getResolutionPlani() const {return ResolutionPlani;}
double ParamMasqueXml::getOrigineAlti() const {return OrigineAlti;}
double ParamMasqueXml::getResolutionAlti() const {return ResolutionAlti;}
const QString& ParamMasqueXml::getGeometrie() const {return Geometrie;}

void ParamMasqueXml::setNameFileMnt(const QString& N) { NameFileMnt = N; }
void ParamMasqueXml::setNameFileMasque(const QString& Na) { NameFileMasque = Na; }
void ParamMasqueXml::setNombrePixels(const QSize& No) { NombrePixels = No; }
void ParamMasqueXml::setOriginePlani(const QPointF& O) { OriginePlani = O; }
void ParamMasqueXml::setResolutionPlani(const QPointF& R) { ResolutionPlani = R; }
void ParamMasqueXml::setOrigineAlti(double Or) { OrigineAlti = Or; }
void ParamMasqueXml::setResolutionAlti(double Re) { ResolutionAlti = Re; }
void ParamMasqueXml::setGeometrie(const QString& G) { Geometrie = G; }
