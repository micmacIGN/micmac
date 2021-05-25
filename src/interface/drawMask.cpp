#if defined Q_WS_WIN 
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "drawMask.h"


using namespace std;


QPointF Pt2dr2QPointF(const Pt2dr& P) { return QPointF(P.x,P.y); }
Pt2dr QPointF2Pt2dr(const QPointF& P) { return Pt2dr(P.x(),P.y()); }
qreal realDistance2(const Pt2dr& P) { return P.x*P.x+P.y*P.y; }
qreal realDistance(const Pt2dr& P) { return sqrt(realDistance2(P)); }

QPoint QPointF2QPoint(const QPointF& P) { return QPoint(P.x(),P.y()); }
QPointF QPoint2QPointF(const QPoint& P) { return QPointF(P.x(),P.y()); }
qreal realDistance2(const QPointF& P) { return P.x()*P.x()+P.y()*P.y(); }
qreal realDistance(const QPointF& P) { return sqrt(realDistance2(P)); }


//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//classes mères pour toutes les interfaces d'affichage (avec possibilité d'afficher 2 images en même temps)

RenderArea::RenderArea(PaintInterf& parent, const ParamMain& pMain, int N, int n) : 
	QWidget( &parent ),
	fichierImage( QString() ),
	refImage( QImage() ),
	num( N ),
	npos( n ),
	toolMode( Move ),
	oldToolMode( Move ),
	center( QPointF() ),
	currentScale( 1 ),
	painterScale( 1 ),
	dragStartPosition( QPoint(-1,-1) ),
	dragging( false ),
	parentWindow( &parent ),
	currentSize( QSize(0,0) ),
	done( false ),
	paramMain( &pMain )
{ 
	setBackgroundRole(QPalette::Base);
	setAutoFillBackground(true);
	setContextMenuPolicy(Qt::DefaultContextMenu);
	changeCurrentScale(1);
}

RenderArea::~RenderArea () {}

void RenderArea::display(const QString& imageFile, const QList<std::pair<Pt2dr,Pt2dr> >& pts) {	//images tif non tuilées
	fichierImage = imageFile;
	refImage = QImage(fichierImage);
	if (refImage.isNull()) {
		qMessageBox(this, tr("Read error"),conv(tr("Fail to read image %1.")).arg(fichierImage));
		return;
	}

	center = QPointF(double(refImage.width())/2.0, double(refImage.height())/2.0);

	changeSize();
	done = true;	
}

QSize RenderArea::sizeHint () {
	QSize maxSize = parentWindow->size() - QSize(50,125);	//marges
	if(!maxSize.isValid() || maxSize.width()<1 || maxSize.height()<1) {		
		maxSize = QApplication::desktop()->availableGeometry().size()/2 - QSize(50,100);
	}
	maxSize.setWidth( maxSize.width()/ceil(sqrt((double)num)) );
	int h = ceil(num/ceil(sqrt((double)num)));
	maxSize.setHeight( maxSize.height()/h );	//réduit la taille en fct du nb d'images à afficher
	
	//-- qreal oldPainterScale = painterScale;
	painterScale = 1.0/max(double(refImage.width())/double(maxSize.width()),double(refImage.height())/double(maxSize.height()));
	//currentScale *= painterScale/oldPainterScale;

	currentSize = refImage.size()*painterScale;
	//if (newSize==currentSize) return newSize;
	setCursor(Qt::WaitCursor);
	
	//émission des paramètres d'affichage
//	getParamDisplay();

	setCursor(Qt::ArrowCursor);
//	parentWidget()->resize(currentSize.width()*ceil(sqrt(num)), currentSize.height()*h);	//renderBox
	parentWidget()->resize(parentWindow->sizeHint2());	//renderBox
	return currentSize;
}

void RenderArea::changeSize() { 
	resize(sizeHint());
}

void RenderArea::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event);
	changeSize();
}

void RenderArea::paintEvent(QPaintEvent*) {
//appelée entre autres à chaque update()
//redessine l'image de fond
	QRect rectangleSource;
	QRect rectangleDestination(0,0,currentSize.width(),currentSize.height());
	changeCurrentScale(currentScale);
	qreal w=double(refImage.width())/currentScale;
	qreal h=double(refImage.height())/currentScale;
	if (center.x()-w/2.0<0)
		center.setX(w/2);
	else if (center.x()+w/2.0>refImage.width())
		center.setX(refImage.width()-w/2);
	if (center.y()-h/2.0<0)
		center.setY(h/2);
	else if (center.y()+h/2.0>refImage.height())
		center.setY(refImage.height()-h/2);
	rectangleSource = QRect (center.x()-w/2.0,center.y()-h/2.0,w,h);

	QPainter painter(this);
     	painter.drawImage(rectangleDestination, refImage, rectangleSource);
	painter.end();
}

void RenderArea::changeCurrentScale(qreal cScale) {
	//if (cScale<min(1.0,painterScale)) cScale=min(1.0,painterScale);
	//if (cScale>10.0*max(1.0,painterScale)) cScale=10.0*max(1.0,painterScale);
	if (cScale<1) cScale=1;
	if (cScale>10) cScale=10;
	currentScale = cScale;	
}

QPointF RenderArea::transfo(QPointF P) const {	//transforme le point cliqué en coordonnées souris en point sur l'image refImage
	return QPointF(P-QPointF(currentSize.width(),currentSize.height())/2.0)/currentScale/painterScale + QPointF(center);
}

QPointF RenderArea::transfoInv(QPointF P) const {	//transforme le point sur l'image refImage en point en coordonnées souris 
	return QPointF(P-QPointF(center))*painterScale*currentScale + QPointF(currentSize.width(),currentSize.height())/2.0;
}

QPointF RenderArea::transfo(QPoint P) const {	//transforme le point cliqué en coordonnées souris en point sur l'image refImage
	return transfo(QPointF(P));
}

QPointF RenderArea::transfoInv(QPoint P) const {	//transforme le point sur l'image refImage en point en coordonnées souris 
	return transfoInv(QPointF(P));
}

QRect RenderArea::transfoInv(QRect rect) const {	//transforme le point sur l'image refImage en point en coordonnées souris 
	QPoint P1 = QPointF2QPoint(transfoInv( QPoint( rect.left(), rect.top() ) ));
	QPoint P2 = QPointF2QPoint(transfoInv( QPoint( rect.right(), rect.bottom() ) ));
	return QRect(P1,P2);
}

//rem : le clic souris event->pos() dépend de currentscale uniquement,
// painter dépend de painterScale, center et currentSize aussi
//les points du masque sont les coordonnées sur refImage
// refImage est fixe
// currentscale dépend des outils mais pas de painterScale
// painterScale dépend de la taille de la fenêtre, réglée par l'utilisateur

void RenderArea::mousePressEvent (QMouseEvent* event) { 
	if (event->button() != Qt::LeftButton) return;
	switch (toolMode) {
		case RenderArea::Move:
			dragging=true;
			dragStartPosition = (transfo(event->pos()));
			break;

		case RenderArea::ZoomIn: //+20%
			center = transfo(event->pos());
			changeCurrentScale(currentScale*1.2);
			update();
			break;

		case RenderArea::ZoomOut: //-20%
			center = transfo(event->pos());
			changeCurrentScale(currentScale*0.8);
			update();
			break;
		default: break;
	}
	emit updated(npos-1);
}

void RenderArea::zoomFullClicked() {
	changeCurrentScale(1);
	update();
	emit updated(npos-1);
}

void RenderArea::mouseMoveEvent(QMouseEvent* event) { 
	if ((toolMode!=Move) || !dragging)
		return;
	center = center + dragStartPosition - transfo(event->pos());
	update();
	emit updated(npos-1);
}

void RenderArea::mouseReleaseEvent(QMouseEvent*) { 
	if ((toolMode!=Move) || !dragging)
		return;
	dragging = false;
}

void RenderArea::setToolMode (ToolMode mode) { toolMode=mode; }
RenderArea::ToolMode RenderArea::getToolMode () const { return toolMode; }
bool RenderArea::getDone() const {return done;}
QSize RenderArea::getCurrentSize() const { return currentSize; }
RenderArea::ParamDisplay RenderArea::getParamDisplay() const {
	ParamDisplay paramDisplay;
	paramDisplay.painterScale = painterScale;
	paramDisplay.currentScale = currentScale;
	paramDisplay.center = center;
	paramDisplay.size = currentSize;
	return paramDisplay;
}
bool RenderArea::masqueIsEmpty() const { return true; }
bool RenderArea::undoPointsIsEmpty() const { return true; }
bool RenderArea::noPolygone() const { return true; }
void RenderArea::undoClicked() {}
void RenderArea::redoClicked() {}
QPoint RenderArea::getPoint() const { return QPoint(); }
std::pair<QPoint,QPoint> RenderArea::getSegment() const { return pair<QPoint,QPoint>(QPoint(),QPoint()); }
int RenderArea::getNbPoint() const { return -1; }
bool RenderArea::ptSauvegarde() const { return true; }
void RenderArea::setGradTool (bool use) {}
void RenderArea::updateGradRegul(int n) {}
Tiff_Im* RenderArea::endPolygone () { return 0; }
void RenderArea::setPoint(const QPoint& P) {}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


PaintInterf::PaintInterf(const ParamMain* pMain, Assistant* help, QWidget* parent):
	QDialog( parent ),
	paramMain( pMain ),
	dir( pMain->getDossier() ),
	imageRef( QStringList() ),
	done( false )
{
	toolBar = new QToolBar();

	setWindowModality(Qt::ApplicationModal);
	resize(maximumSizeHint());
	setMinimumSize(minimumSizeHint());
	setMaximumSize(maximumSizeHint());

	statusBar = new QStatusBar;
	fullScreenButton = new QPushButton(QIcon(g_iconDirectory+"windows_fullscreen.png"), QString());
	fullScreenButton->setToolTip(conv(tr("Full screen")));
	fullScreenButton->setMaximumSize(40,34);
	statusBar->addPermanentWidget(fullScreenButton);
	assistant = help;

	mainLayout = new QVBoxLayout;
	mainLayout->addWidget(toolBar,0,Qt::AlignTop | Qt::AlignLeft);
	mainLayout->insertSpacing (1,25);
	mainLayout->insertSpacing (2,maximumSizeHint().height());
	mainLayout->addWidget(statusBar,0,Qt::AlignTop);
	mainLayout->addStretch();
	setLayout(mainLayout);

	setWindowTitle(tr("Mask draw"));
	connect(fullScreenButton, SIGNAL(clicked()), this, SLOT(fullScreenClicked()));

	done = true;
}

PaintInterf::~PaintInterf () {
	for (int i=0; i<renderArea.count(); i++)
		delete renderArea.at(i);
	delete okAct;
	delete dragAct;
	delete zoomInAct;
	delete zoomOutAct;
	delete zoomFullAct; 
	delete helpAct; 
}

QSize PaintInterf::sizeHint() const { return QApplication::desktop()->availableGeometry().size()/2; }
QSize PaintInterf::minimumSizeHint() const { return QApplication::desktop()->availableGeometry().size()/4; }
QSize PaintInterf::maximumSizeHint() const { return QApplication::desktop()->availableGeometry().size(); }
QSize PaintInterf::sizeHint2() const {
	QSize s(0,0);
	for (int i=0; i<renderArea.count(); i++)
		s = QSize( s.width()+renderArea.at(i)->getCurrentSize().width() , max(s.height(),renderArea.at(i)->getCurrentSize().height()) );
	return s;
}

void PaintInterf::display() {
//affichage des renderArea et connections signals/slots
	QHBoxLayout* renderLayout = new QHBoxLayout;
	for (int i=0; i<renderArea.count(); i++) {
		if (i>0) renderLayout->insertSpacing (1,maximumSizeHint().width());
		renderLayout->addWidget(renderArea.at(i),0,Qt::AlignTop | Qt::AlignLeft);
		if (i>0) renderLayout->insertSpacing (3,maximumSizeHint().width());
		//connect(renderArea.at(i), SIGNAL(sigParam(RenderArea::ParamDisplay,int)), this, SLOT(getParamImg(RenderArea::ParamDisplay,int)));
		connect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)), Qt::QueuedConnection);
	}
	renderLayout->addStretch();
	renderBox = new QGroupBox;
	//renderBox->resize(sizeHint());
	renderBox->setLayout(renderLayout);
	mainLayout->insertWidget(2,renderBox,0,Qt::AlignTop | Qt::AlignLeft);
}

void PaintInterf::createActions() {
	okAct = new QAction(QIcon(g_iconDirectory+"linguist-check-on.png"), tr("&Ok"), this);
	connect(okAct, SIGNAL(triggered()), this, SLOT(okClicked()));

	dragAct = new QAction(QIcon(g_iconDirectory+"cursor-openhand.png"), conv(tr("Image moving")), this);
	connect(dragAct, SIGNAL(triggered()), this, SLOT(dragClicked()));

	zoomInAct = new QAction(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"), tr("&Zoom in"), this);
	connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomInClicked()));

	zoomOutAct = new QAction(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic.png"), conv(tr("Zoom out")), this);
	connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOutClicked()));

	zoomFullAct = new QAction(QIcon(g_iconDirectory+"cursor-sizeall.png"), conv(tr("Scale to Window")), this);
	for (int i=0; i<renderArea.count(); i++)
		connect(zoomFullAct, SIGNAL(triggered()), renderArea.at(i), SLOT(zoomFullClicked()));

	helpAct = new QAction(QIcon(g_iconDirectory+"linguist-check-off.png"), tr("Help"), this);
	connect(helpAct, SIGNAL(triggered()), this, SLOT(helpClicked()));
}

void PaintInterf::createToolBar() {
	toolBar->addAction(okAct);
	toolBar->addSeparator ();
	toolBar->addAction(dragAct);
	toolBar->addAction(zoomInAct);
	toolBar->addAction(zoomOutAct);
	toolBar->addAction(zoomFullAct);
	toolBar->addSeparator ();
	toolBar->addAction(helpAct);
	updateToolBar(RenderArea::Move);
}

void PaintInterf::updateToolBar(const RenderArea::ToolMode& mode) {
	dragAct->setChecked(false);  
	zoomInAct->setChecked(false);  
	zoomOutAct->setChecked(false);  
	statusBar->clearMessage ();

	dragAct->setCheckable (true);
	zoomInAct->setCheckable (true);
	zoomOutAct->setCheckable (true);

	switch (mode) {
		case RenderArea::Move:
			dragAct->setChecked(true);  
			break;

		case RenderArea::ZoomIn:
			zoomInAct->setChecked(true); 
			break;

		case RenderArea::ZoomOut:
			zoomOutAct->setChecked(true); 
			break;
		default: break;
	}
	for (int i=0; i<renderArea.count(); i++)
		renderArea.at(i)->setToolMode(mode);
}

void PaintInterf::dragClicked() {updateToolBar(RenderArea::Move);}
void PaintInterf::zoomInClicked() {updateToolBar(RenderArea::ZoomIn);}
void PaintInterf::zoomOutClicked() {updateToolBar(RenderArea::ZoomOut);}
void PaintInterf::helpClicked() {} 

void PaintInterf::okClicked() {
	accept();
	hide();
}

void PaintInterf::fullScreenClicked() {
	if (fullScreenButton->toolTip()==tr("Common size")) {
		//on remet à la taille normale
		resize(sizeHint());
		move(maximumSizeHint().width()/4, maximumSizeHint().height()/4);
		fullScreenButton->setToolTip(conv(tr("Full screen")));
	} else {
		//on remet en plein écran
		resize(maximumSizeHint());
		fullScreenButton->setToolTip(tr("Common size"));	
	}
}

RenderArea::ParamDisplay PaintInterf::getPosImg(int num) const {
//permet de partager les paramètres d'affichage d'une renderArea à l'autre
	int n2 = 2-num;
	//RenderArea::ParamDisplay posNum = renderArea.at(num-1)->getParamDisplay();
	RenderArea::ParamDisplay posN2 = renderArea.at(n2)->getParamDisplay();
	//posN2.origine = posN2.origine + renderArea.at(n2)->pos() - renderArea.at(num-1)->pos();
	posN2.origine = renderArea.at(n2)->pos() - renderArea.at(num-1)->pos();
	return posN2;
}

void PaintInterf::updateAll(int pos) {
//à chaque modification d'une renderArea, met aussi les autres à jour, pour déplacer les segments
	/*for (int i=0; i<renderArea.count(); i++) {
		if (i==pos) continue;
		disconnect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)));
		renderArea.at(i)->update();
		connect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)));
	}*/
	//for (int i=0; i<renderArea.count(); i++) disconnect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)));
	for (int i=0; i<renderArea.count(); i++) disconnect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)));
	for (int i=0; i<renderArea.count(); i++) {
		if (i==pos) continue;
		renderArea.at(i)->update();
	}
	for (int i=0; i<renderArea.count(); i++) connect(renderArea.at(i), SIGNAL(updated(int)), this, SLOT(updateAll(int)), Qt::QueuedConnection);
}

bool PaintInterf::getDone() const {return done;}
bool PaintInterf::masqueIsEmpty() const { return true; }
void PaintInterf::setLastPoint(int n) {}
pair<QPoint,QPoint> PaintInterf::getSegment() const { return pair<QPoint,QPoint>(QPoint(),QPoint()); }
int PaintInterf::getNbPoint(int n) const {return -1;}
int PaintInterf::maxSpinBox() const { return 0; }
Tiff_Im* PaintInterf::getMaskImg() { return 0; }


//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//affichage des points homologues

VisuHomologues::VisuHomologues(VueHomologues& parent, const ParamMain& pMain, int pos) : RenderArea(parent, pMain, 2, pos), points(0), num(0), marquise(QRect()) {}	//2 images

VisuHomologues::~VisuHomologues () {}

void VisuHomologues::display(const QString& imageFile, const QList<std::pair<Pt2dr,Pt2dr> >& pts) {
	points = &pts;
	RenderArea::display(imageFile, pts);
	update();
}

void VisuHomologues::mousePressEvent(QMouseEvent* event) {
	if (toolMode==RenderArea::Filtre && event->button()==Qt::LeftButton) {
		dragging=true;
		dragStartPosition = event->pos();
	}
	RenderArea::mousePressEvent(event);
}

void VisuHomologues::mouseMoveEvent(QMouseEvent* event) {
	if (toolMode==RenderArea::Filtre && dragging) {
		marquise = QRect(QPointF2QPoint(dragStartPosition), event->pos()).normalized();
		update();	//dessin de la marquise
		emit updated(npos-1);
	}
	RenderArea::mouseMoveEvent(event);
}

void VisuHomologues::mouseReleaseEvent(QMouseEvent* event) {
	if (toolMode==RenderArea::Filtre && dragging) {
		dragging = false;
		if (dragStartPosition==event->pos()) {
			dynamic_cast<VueHomologues*>(parentWindow)->supprCouples(transfo(event->pos()), npos, 10.0/currentScale/painterScale);	//suppression du point le plus proche
		} else {
			dynamic_cast<VueHomologues*>(parentWindow)->supprCouples(QRect(QPointF2QPoint(transfo(dragStartPosition)),QPointF2QPoint(transfo(event->pos()))).normalized(), npos); //suppression des points à l'intérieur de la marquise
			update();	//suppression du dessin de la marquise
		}
	}
	RenderArea::mouseReleaseEvent(event);
}

void VisuHomologues::paintEvent(QPaintEvent* event) {	
	RenderArea::paintEvent(event);
	
	//dessin des points
	ParamDisplay posImg2 = parentWindow->getPosImg(npos); //récupération des paramètres de l'autre image
	QPainter painter(this);	//=> points à dessiner en coordonnées souris
		//points
		painter.setRenderHint(QPainter::Antialiasing, true);
		QPen pen(QColor(255,0,0,255));
		pen.setWidth(1);
		painter.setPen(pen);
		for (int i=0; i<points->count(); i++) {
			//récupération du segment
			QPoint P1(points->at(i).first.x, points->at(i).first.y);
			QPoint P2(points->at(i).second.x, points->at(i).second.y);
			if (npos==2) {	//on inverse les rôles
				QPoint P = P1;
				P1 = P2;
				P2 = P;
			}
			//coordonnées souris
			QPoint p1 = QPointF2QPoint(transfoInv(QPointF(P1)));
			QPointF p2bF = (QPointF(P2)-QPointF(posImg2.center))*posImg2.currentScale*posImg2.painterScale + QPointF(posImg2.size.width(),posImg2.size.height())/2.0;	//P2 dans l'image 2
			QPoint p2b(p2bF.x(),p2bF.y());
			QPoint p2 = QPointF2QPoint((P2-posImg2.center)*posImg2.currentScale*posImg2.painterScale + QPoint(posImg2.size.width(),posImg2.size.height())/2 + posImg2.origine);	//P2 dans l'image 1
			//points à afficher
			QRect rectangleSource(0, 0, currentSize.width(), currentSize.height());
			if (!rectangleSource.contains(p1)) continue;
			QRect rectangleSource2(0, 0, posImg2.size.width(), posImg2.size.height());
			if (!rectangleSource2.contains(p2b)) continue;
			//point intermédiaire
			if (p1.x()==p2.x()) continue;
			int x = (npos==2)? 0 : currentSize.width();
			QPoint p(x, p1.y()+double(p1.y()-p2.y())/double(p1.x()-p2.x())*(x-p1.x()));
			painter.drawLine(p1, p);
		}
		//marquise
		if (toolMode==RenderArea::Filtre && dragging) {
			pen.setColor(QColor(0,0,255,255));
			painter.setPen(pen);
			painter.drawRect(marquise);
		}
	painter.end();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


VueHomologues::VueHomologues(const ParamMain* pMain, Assistant* help, QWidget* parent) : 
	PaintInterf( pMain,help,parent ),
	couples( QList<LiaisCpl>() ),	//fichiers de couples (pb : homol a moins de fichiers que ceux demandés)
	nomFichiers( QList<ElSTDNS string>() ),
	undoCouples( QList<LiaisCpl>() ),
	redoCouples( QList<LiaisCpl>() ),
	changed( false )
{
	//paramètres
	done = false;
	//points de liaison
	cTplValGesInit<string>  aTpl;
	char** argv = new char*[1];
	char c_str[] = "rthsrth";
	argv[0] = new char[strlen( c_str )+1];
	strcpy( argv[0], c_str );
	cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, dir.toStdString(), aTpl );
	const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");
	for (int aK=0; aK<signed(aVN->size()) ; aK++) {
		couples.push_back(LiaisCpl());
		nomFichiers.push_back((*aVN)[aK]);
		LiaisCpl* couple = &(couples[couples.count()-1]);
		//bon couple
		pair<string,string>  aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
		if (signed(aPair.first.find("_init"))!=-1 || signed(aPair.second.find("_init"))!=-1) continue;
		if (signed(aPair.first.find("_filtre"))!=-1 || signed(aPair.second.find("_filtre"))!=-1) continue;
		//extraction des points
		ElPackHomologue aPack = ElPackHomologue::FromFile(dir.toStdString()+(*aVN)[aK]);
		if (aPack.size()==0) continue;
		for (ElPackHomologue::const_iterator  itH=aPack.begin(); itH!=aPack.end() ; itH++)
		{				
			Pt2dr pt1(itH->P1());
			Pt2dr pt2(itH->P2());
			couple->pointsLiaison.push_back( pair<Pt2dr,Pt2dr>(pt1,pt2) );
		}
		if (couple->pointsLiaison.count()!=0) {
			couple->image1 = QString(aPair.first.c_str());
			couple->image2 = QString(aPair.second.c_str());
		} else {
			couples.removeAt(couples.count()-1);
		}
	}
	delete [] argv[0];
	delete [] argv;
	delete mICNM;
	if (couples.count()==0) {
		qMessageBox(this, tr("Read error"), conv(tr("No tie-points found.")));
		return;
	}

	//affichage des images
	for (int i=0; i<2; i++) {
		renderArea.push_back( new VisuHomologues(*this, *paramMain, i+1) );
	}
	display();	//PaintInterf::display()

	filtrBar = new QToolBar();
	createActions();
	createToolBar();
	mainLayout->insertWidget(1,filtrBar,0,Qt::AlignTop);
	filtrBar->hide();

	//choix des images
	QLabel* label1 = new QLabel(tr("Image 1"));
	liste1 = new QComboBox;
	for (int i=0; i<paramMain->getCorrespImgCalib().count(); i++) {
		liste1->addItem(paramMain->getCorrespImgCalib().at(i).getImageTif());
	}
	QLabel* label2 = new QLabel(tr("Image 2"));
	liste2 = new QComboBox;

	QHBoxLayout *listesLayout = new QHBoxLayout;
	listesLayout->addWidget(label1);
	listesLayout->addWidget(liste1);
	listesLayout->addWidget(label2);
	listesLayout->addWidget(liste2);
	listesLayout->addStretch();

	QGroupBox* listesBox = new QGroupBox;
	listesBox->setLayout(listesLayout);
	mainLayout->insertWidget(5,listesBox,0,Qt::AlignTop | Qt::AlignLeft);
	mainLayout->addStretch();

	connect(liste1, SIGNAL(currentIndexChanged(int)), this, SLOT(liste1Clicked()));
	connect(liste2, SIGNAL(currentIndexChanged(int)), this, SLOT(liste2Clicked()));

	setWindowTitle(tr("Tie-point view"));

	liste1Clicked();
	done = true;
}

VueHomologues::~VueHomologues () {
	delete filtrAct;
	delete marquiseAct; 
	delete undoAct; 
	delete redoAct; 
	delete saveAct; 
}

void VueHomologues::createActions() {
	PaintInterf::createActions();
	filtrAct = new QAction(QIcon(g_iconDirectory+"bin.png"), tr("&Display toolbar for deleting mismatches manually"), this);
	connect(filtrAct, SIGNAL(triggered()), this, SLOT(displayFiltreClicked()));

	//barre d'outil de filtrage
	marquiseAct = new QAction(QIcon(g_iconDirectory+"cursor-arrow.png"), tr("&Tie-point selection and removal (make a frame or click)"), this);
	connect(marquiseAct, SIGNAL(triggered()), this, SLOT(doMarquiseClicked()));

	undoAct = new QAction(QIcon(g_iconDirectory+"linguist-editundo.png"), conv(tr("Restore deleted point(s)")), this);
	connect(undoAct, SIGNAL(triggered()), this, SLOT(undoClicked()));

	redoAct = new QAction(QIcon(g_iconDirectory+"linguist-editredo.png"), tr("&Delete restored matches"), this);
	connect(redoAct, SIGNAL(triggered()), this, SLOT(redoClicked()));

	zoomOutAct = new QAction(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic.png"), conv(tr("Zoom out")), this);
	connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOutClicked()));

	saveAct = new QAction(QIcon(g_iconDirectory+"linguist-filesave.png"), conv(tr("Save point removal in files")), this);
	connect(saveAct, SIGNAL(triggered()), this, SLOT(saveClicked()));
	changed = false;
}

void VueHomologues::createToolBar() {
	PaintInterf::createToolBar();
	toolBar->addSeparator ();
	toolBar->addAction(filtrAct);

	//barre d'outils pour le filtrage des points homologues
	filtrBar->addAction(marquiseAct);
	filtrBar->addAction(undoAct);
	filtrBar->addAction(redoAct);
	filtrBar->addAction(saveAct);
}

void VueHomologues::updateToolBar(const RenderArea::ToolMode& mode) {
	filtrAct->setCheckable (true);
	marquiseAct->setCheckable (true);
	undoAct->setCheckable(false);  
	redoAct->setCheckable(false);  
	saveAct->setCheckable(false); 

	undoAct->setEnabled (undoCouples.count()>0);
	redoAct->setEnabled (redoCouples.count()>0);
	saveAct->setEnabled (changed);

	marquiseAct->setChecked(mode==RenderArea::Filtre);  
	PaintInterf::updateToolBar(mode);
}

void VueHomologues::liste1Clicked() {
	//affichage des la liste des images 2 (couples) à chaque changement de l'image 1
	liste2->clear();
	QString img1 = liste1->itemText(liste1->currentIndex());
	for (int i=0; i<couples.count(); i++) {
		if (couples.at(i).image1!=img1) continue;
		liste2->addItem(couples.at(i).image2);
	}	
	liste2Clicked();
}

void VueHomologues::liste2Clicked() {
	if (liste2->count()==0) return;	//liste1Clicked() : liste2->clear(); => liste2Clicked()
	//modifie la visuArea à chaque changement de l'image 2
		//lecture des images
	QString img1( dir+paramMain->convertTifName2Couleur(liste1->itemText(liste1->currentIndex())) );
	QString img2( dir+paramMain->convertTifName2Couleur(liste2->itemText(liste2->currentIndex())) );

		//récupération des points
	int N = -1;
	for (int i=0; i<couples.count(); i++) {
		if (couples.at(i).image1==liste1->itemText(liste1->currentIndex()) && couples.at(i).image2==liste2->itemText(liste2->currentIndex()))
			N = i;
	}

		//affichage
	renderArea.at(0)->display(img1, couples.at(N).pointsLiaison);
	renderArea.at(1)->display(img2, couples.at(N).pointsLiaison);
}

void VueHomologues::helpClicked() {assistant->showDocumentation(assistant->pageVueHomologues); } 

void VueHomologues::displayFiltreClicked() {
	if (filtrAct->isChecked()) filtrBar->show();
	else filtrBar->hide();
	updateToolBar(renderArea.at(0)->getToolMode());
}

void VueHomologues::doMarquiseClicked() {
	updateToolBar(RenderArea::Filtre);
}

void VueHomologues::supprCouples(const QPointF& P, int vue, double rayon) {
	QString img1( liste1->itemText(liste1->currentIndex()) );
	QString img2( liste2->itemText(liste2->currentIndex()) );
	int idx = couples.indexOf(LiaisCpl(img1,img2));
	QList<std::pair<Pt2dr,Pt2dr> >* pLC = &couples[idx].pointsLiaison;
	QList<QList<std::pair<Pt2dr,Pt2dr> >::iterator> ASupprimer;
	double d = numeric_limits<double>::max();
	for(QList<std::pair<Pt2dr,Pt2dr> >::iterator itP=pLC->begin(); itP!=pLC->end(); itP++) {
		Pt2dr P2 = (vue==1)? itP->first : itP->second;
		double dist = realDistance2(Pt2dr2QPointF(P2)-P);
		if (dist<d) {
			ASupprimer.clear();
			d = dist;
		}
		if (dist<=d) {
			ASupprimer.push_back(itP);
		}
	}
	if (d>rayon*rayon) {
		qMessageBox(this, tr("Warning"), conv(tr("No points deleted.")));
		return;
	}
	if (ASupprimer.count()!=1) {
		qMessageBox(this, tr("Warning"), conv(tr("Conflict between %1 points.\nNo points deleted.").arg(ASupprimer.count())));
		return;
	}
	supprCouples(img1, img2, pLC, ASupprimer);
}

void VueHomologues::supprCouples(const QRect& R, int vue) {
	QString img1( liste1->itemText(liste1->currentIndex()) );
	QString img2( liste2->itemText(liste2->currentIndex()) );
	int idx = couples.indexOf(LiaisCpl(img1,img2));
	QList<std::pair<Pt2dr,Pt2dr> >* pLC = &couples[idx].pointsLiaison;
	QList<QList<std::pair<Pt2dr,Pt2dr> >::iterator> ASupprimer;
	for(QList<std::pair<Pt2dr,Pt2dr> >::iterator  itP=pLC->begin(); itP!=pLC->end(); itP++) {
		if ( ( vue==1 && R.contains(QPointF2QPoint(Pt2dr2QPointF(itP->first))) )
		  || ( vue==2 && R.contains(QPointF2QPoint(Pt2dr2QPointF(itP->second))) ) )
			ASupprimer.push_back(itP);
	}	
	if (ASupprimer.count()==0) {
		qMessageBox(this, tr("Warning"), conv(tr("No points deleted.")));
		return;
	}
	supprCouples(img1, img2, pLC, ASupprimer);
}

void VueHomologues::supprCouples(const QString& img1, const QString& img2, QList<std::pair<Pt2dr,Pt2dr> >* couples, const QList<QList<std::pair<Pt2dr,Pt2dr> >::iterator>& ASupprimer) {
	undoCouples.push_back(LiaisCpl(img1,img2));
	for (int idx=ASupprimer.count()-1; idx>-1; idx--) {
		undoCouples.last().pointsLiaison.push_back(*(ASupprimer.at(idx)));
		couples->erase(ASupprimer.at(idx));
	}
	redoCouples.clear();
	renderArea.at(0)->display( dir+paramMain->convertTifName2Couleur(img1) , *couples);
	renderArea.at(1)->display( dir+paramMain->convertTifName2Couleur(img2) , *couples);
	changed = true;
	updateToolBar(RenderArea::Filtre);
}

void VueHomologues::undoClicked() {
	if (undoCouples.count()==0) return;
	//réaffiche les points qui viennent d'être supprimés
	QString img1 = undoCouples.last().image1;
	QString img2 = undoCouples.last().image2;
	int idx = couples.indexOf(LiaisCpl(img1,img2));
	couples[idx].pointsLiaison << undoCouples.last().pointsLiaison;
	//enregistrement
	redoCouples.push_back(undoCouples.last());
	undoCouples.pop_back();
	//affichage
	renderArea.at(0)->display( dir+paramMain->convertTifName2Couleur(img1), couples.at(idx).pointsLiaison);
	renderArea.at(1)->display( dir+paramMain->convertTifName2Couleur(img2), couples.at(idx).pointsLiaison);
	changed = true;
	updateToolBar(renderArea.at(0)->getToolMode());
}

void VueHomologues::redoClicked() {
	if (redoCouples.count()==0) return;
	//résupprime les points qui viennent d'être réaffichés
	QString img1 = redoCouples.last().image1;
	QString img2 = redoCouples.last().image2;
	int idx = couples.indexOf(LiaisCpl(img1,img2));
	int nb = redoCouples.last().pointsLiaison.count();
	for (int i=0; i<nb; i++)
		couples[idx].pointsLiaison.pop_back();
	//enregistrement
	undoCouples.push_back(redoCouples.last());
	redoCouples.pop_back();
	//affichage
	renderArea.at(0)->display( dir+paramMain->convertTifName2Couleur(img1), couples.at(idx).pointsLiaison);
	renderArea.at(1)->display( dir+paramMain->convertTifName2Couleur(img2), couples.at(idx).pointsLiaison);
	changed = true;
	updateToolBar(renderArea.at(0)->getToolMode());
}

void VueHomologues::saveClicked() {
	QApplication::setOverrideCursor( Qt::WaitCursor );
	for (QList<LiaisCpl>::const_iterator undoIt=undoCouples.begin(); undoIt!=undoCouples.end(); undoIt++) {
		QString img1 = undoIt->image1;
		QString img2 = undoIt->image2;
		int idx = couples.indexOf(LiaisCpl(img1,img2));
		const QList<std::pair<Pt2dr,Pt2dr> >* pLC = &couples[idx].pointsLiaison;
		
		ElPackHomologue  aPack;
		for(QList<std::pair<Pt2dr,Pt2dr> >::const_iterator  itP=pLC->begin(); itP!=pLC->end(); itP++) {
			ElCplePtsHomologues aCple (itP->first,itP->second);
			aPack.Cple_Add(aCple);
		}

		ElSTDNS string s = paramMain->getDossier().toStdString()+nomFichiers.at(idx);
		aPack.StdPutInFile(s);	//supprimer aussi dans -filtre et _init => dans ce cas il faut les lire et rechercher les bons couples
	}
	changed = false;
	updateToolBar(renderArea.at(0)->getToolMode());
	QApplication::restoreOverrideCursor();
	qMessageBox(this, tr("Information"), conv(tr("Points successfully removed from files.")));
}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//


DrawArea::DrawArea(DrawInterf& parent, const ParamMain& pMain, const QString& imageFile, int N, int n):
	RenderArea( parent, pMain,N,n ),
	refImageClean( QImage() ),
	masque( QList<Polygone>() ),
	sauvegarde( QList<Polygone>() ),
	updateBuffer( 0 ),
	undoCompteur( QList<int>() ),
	redoCompteur( QList<int>() ),
	withGradTool( false ),
	maxPoint( -1 )
{
	withGradTool = false;

	toolMode=Draw;
	oldToolMode=Draw;

	display(imageFile);	//RenderArea::display(imageFile)
	refImageClean = refImage;
}

DrawArea::~DrawArea () {}

//rem : le clic souris event->pos() dépend de currentscale uniquement,
// painter dépend de painterScale, center et image.size() aussi
//les points du masque sont les coordonnées sur image, ceux sur masque sont les coordonnées sur refImage
// refImage est fixe
// currentscale dépend des outils mais pas de painterScale
// painterScale dépend de la taille de la fenêtre, réglée par l'utilisateur

void DrawArea::mousePressEvent (QMouseEvent * event) {
	RenderArea::mousePressEvent(event); 
	if (event->button() == Qt::LeftButton) {
		switch (toolMode) {
			case RenderArea::Move:
			case RenderArea::ZoomIn:
			case RenderArea::ZoomOut:
				return;
			default : break;
		}
		//ajout du point au masque ou au segment
		if (maxPoint!=-1 && masque.count()>0 && masque.at(0).getQpolygon().count()==maxPoint) return; //il y a déjà le maximum de points possibles (cas segment)
		int rad = (1.0 / 2.0) + 2;
		
		QPolygon* currentPolygone = continuePolygone (masque,rad);
		QPoint endPoint=QPointF2QPoint(transfo(event->pos()));
	
		if (currentPolygone->size()>0) {	//suite du polygone => 1 segment
			QPoint lastPoint=currentPolygone->at(currentPolygone->count()-1);
			if (endPoint==lastPoint) return;
			if (!withGradTool || (endPoint-lastPoint).manhattanLength()<=1 || dynamic_cast<PaintInterfPlan*>(parentWindow)==0) {	
				currentPolygone->push_back(endPoint);
				undoCompteur.push_front(1);
				update(transfoInv(QRect(lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad)));
			}
		} else {	//nouveau polygone => 1 point
			currentPolygone->push_back(endPoint);
			undoCompteur.push_front(1);
			update(transfoInv(QRect(endPoint-QPoint(rad,rad), endPoint+QPoint(rad,rad))));
		}
		parentWindow->updateToolBar(toolMode);
	} 
}

QPolygon* DrawArea::continuePolygone (QList<Polygone>& conteneur, int rad) {
	bool addone = false;
	if (conteneur.size()==0) addone = true;
	else {
		ToolMode otherTool = (toolMode==Draw) ? Cut : Draw;
		Polygone* polyg = &(conteneur)[conteneur.count()-1];
		ToolMode oldTmode = polyg->getTmode();
		if (oldTmode==otherTool && !(polyg->getFerme())) {	//si le polygone précédent était utilisé avec l'autre outil et n'a pas été fermé, on le supprime
			QRect rect = polyg->getQpolygon().boundingRect();
			polyg->modifQpolygon().clear();
			polyg->setTmode(toolMode);
			if (rad!=-1)
				update(rect.normalized().adjusted(-rad, -rad, +rad, +rad));
		}
		else if (polyg->getFerme()) addone = true;
	}
	if (addone) {
		Polygone newPolygone(toolMode,false);
		conteneur.push_back(newPolygone);
	}
	return &(masque[masque.count()-1].modifQpolygon());
}

void DrawArea::paintEvent(QPaintEvent* event) {
	//dessin de l'image
	RenderArea::paintEvent(event);
	if (masque.count()==0 || masque.at(0).getQpolygon().count()==0) return;
	if (maxPoint==1) return; //cas particulier du segment à cheval sur 2 images

	QList<Polygone> masqueAlEchelle;
	changeScale(masque, masqueAlEchelle);	///currentScale ?

	QPainter painter(this);	 //on ne peut pas dessiner les polygones avec QPainter(refImage) car on veut que l'épaisseur de la ligne reste fixe (donc pas dans updateAllPolygones())
		//le dernier polygone est-il fermé ?
		int itend=masque.count();
		if (!(masque.at(masque.count()-1).getFerme()))
			--itend;

		//dessin polygones fermés sans les contours
		painter.setPen(Qt::NoPen);
		QBrush brush(Qt::SolidPattern);
		QRect rect(center.x()*painterScale*currentScale-currentSize.width()/2,center.y()*painterScale*currentScale-currentSize.height()/2,currentSize.width(),currentSize.height());
		brush.setTextureImage(refImageClean.scaled(currentSize*currentScale).copy(rect));
		for (int it=0; it<itend; it++) {
			if (masque.at(it).getTmode()==Draw) {
				painter.setBrush(QBrush(QColor(0,255,0,255),Qt::Dense6Pattern));
				//painter.setBrush(QBrush(QColor(0,255,0,20),Qt::SolidPattern));
				painter.drawPolygon(masqueAlEchelle.at(it).getQpolygon(), Qt::WindingFill);
			} else if (masque.at(it).getTmode()==Cut) {
				painter.setBrush(brush);
				painter.drawPolygon(masqueAlEchelle.at(it).getQpolygon(), Qt::WindingFill);
			} 
		}

		//dessin des polygones non fermés par leur contour
		if (itend!=masque.count()) {	
			const Polygone* p = &(masqueAlEchelle[itend]);
			painter.setBrush(Qt::NoBrush);
			QColor color = (p->getTmode()==Draw)? QColor(0,255,0,255) : QColor(255,0,0,255);
			QPen pen(color);
			pen.setWidth(1);
			painter.setPen(pen);
			if (p->getQpolygon().count()==1) {
				QPoint P = p->getQpolygon().at(0);
				//painter.drawPoint(p->getQpolygon().at(0));	//on dessine plutôt une croix pour mieux le voir
				painter.drawLine(P+QPoint(-10,-10),P+QPoint(10,10));
				painter.drawLine(P+QPoint(-10,10),P+QPoint(10,-10));
				if (dynamic_cast<PaintInterfAppui*>(parentWindow)!=0) {
					painter.drawLine(P+QPoint(-1,-1), P+QPoint(1,1));
					painter.drawLine(P+QPoint(-1,1), P+QPoint(1,-1));
				}
			} else if (p->getQpolygon().count()>0) {
				for (int it1=0; it1<p->getQpolygon().count()-1; it1++) {
					painter.drawLine(p->getQpolygon().at(it1), p->getQpolygon().at(it1+1));	
				}
			}
		}
	painter.end();
}

void DrawArea::changeScale(const QList<Polygone>& conteneur, QList<Polygone>& conteneurAlEchelle) const {
//remet les dessins à l'échelle dans conteneurAlEchelle (conteneur est à l'échelle initiale)
	if (conteneur.count()==0 || conteneur.at(0).getQpolygon().count()==0)  return;
	conteneurAlEchelle.clear();
	for (int i=0; i<conteneur.count(); i++) {
		conteneurAlEchelle.push_back(conteneur.at(i).clone());
		for (int j=0; j<conteneur.at(i).getQpolygon().count(); j++)
			conteneurAlEchelle[i].modifQpolygon().push_back(QPointF2QPoint(transfoInv(conteneur.at(i).getQpolygon().at(j))));
	}
}

bool DrawArea::masqueIsEmpty() const {return (masque.size()==0);}	//undo->setEnabled()
bool DrawArea::undoPointsIsEmpty() const {return (sauvegarde.size()==0);}	//redo->setEnabled()

//----------------------------------------------------------------------------------------------------------------------------------------//

DrawArea::Polygone::Polygone() : tmode(ToolMode()), ferme(bool()), qpolygon(QPolygon()) {}
DrawArea::Polygone::Polygone(const DrawArea::Polygone& polygone) { copie(polygone); }
DrawArea::Polygone::Polygone(const ToolMode& t, bool f, const QPolygon& q) : tmode(t), ferme(f), qpolygon(q) {}
DrawArea::Polygone::~Polygone() {}

DrawArea::Polygone& DrawArea::Polygone::operator=(const DrawArea::Polygone& polygone) {
	if (this!=&polygone) copie(polygone);
	return *this;
}
DrawArea::Polygone DrawArea::Polygone::clone() const {
	Polygone P;
	P.setTmode(getTmode());
	P.setFerme(getFerme());
	return P;
}

void DrawArea::Polygone::copie(const DrawArea::Polygone& polygone) {
	tmode = polygone.getTmode();
	ferme = polygone.getFerme();
	qpolygon = polygone.getQpolygon();
}

const RenderArea::ToolMode& DrawArea::Polygone::getTmode() const { return tmode; }
bool DrawArea::Polygone::getFerme() const { return ferme; }
const QPolygon& DrawArea::Polygone::getQpolygon() const { return qpolygon; }

void DrawArea::Polygone::setTmode(const RenderArea::ToolMode& t) { tmode = t; }
void DrawArea::Polygone::setFerme(bool f) { ferme = f; }
void DrawArea::Polygone::setQpolygon(const QPolygon& q) { qpolygon = q; }
QPolygon& DrawArea::Polygone::modifQpolygon() { return qpolygon; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


DrawInterf::DrawInterf(const ParamMain* pMain, Assistant* help, QWidget* parent) : PaintInterf(pMain, help, parent), num(0) {}

DrawInterf::~DrawInterf () {
	delete cutAct;
	delete drawAct;
	delete undoAct;
	delete redoAct;
	delete clearAct;
}

void DrawInterf::createActions() {
	PaintInterf::createActions();
	drawAct = new QAction(QIcon(g_iconDirectory+"designer-edit-resources-button.png"), tr("&Draw mask"), this);
	connect(drawAct, SIGNAL(triggered()), this, SLOT(drawClicked()));

	cutAct = new QAction(QIcon(g_iconDirectory+"linguist-editcut.png"), conv(tr("Cut the mask")), this);
	connect(cutAct, SIGNAL(triggered()), this, SLOT(cutClicked()));

	undoAct = new QAction(QIcon(g_iconDirectory+"linguist-editundo.png"), tr("&Undo last point"), this);
	redoAct = new QAction(QIcon(g_iconDirectory+"linguist-editredo.png"), conv(tr("Restore last point")), this);

	clearAct = new QAction(QIcon(g_iconDirectory+"qmessagebox-crit.png"), tr("&Clear all points"), this);
}

void DrawInterf::createToolBar() {
	PaintInterf::createToolBar();
	toolBar->addSeparator ();
	toolBar->addAction(drawAct);
	toolBar->addAction(cutAct);
	toolBar->addSeparator ();
	toolBar->addAction(undoAct);
	toolBar->addAction(redoAct);
	toolBar->addAction(clearAct);
	updateToolBar(DrawArea::Draw);
}

void DrawInterf::updateToolBar(const RenderArea::ToolMode& mode) {
	drawAct->setChecked(false);  
	cutAct->setChecked(false);  
	statusBar->clearMessage ();

	drawAct->setCheckable (true);
	cutAct->setCheckable (true);

	cutAct->setEnabled(false);
	undoAct->setEnabled(false);
	redoAct->setEnabled(false);
	clearAct->setEnabled(false);
	for (int i=0; i<renderArea.count(); i++) {
		if (!renderArea[i]->masqueIsEmpty()) {
			undoAct->setEnabled(true);
			clearAct->setEnabled(true);
		}
		if (!renderArea[i]->undoPointsIsEmpty())
			redoAct->setEnabled(true);
		if (!renderArea[i]->noPolygone())
			cutAct->setEnabled(true);
	}

	switch (mode) {//ajouter le changement d'icône quand les boutons sont sélectionnés
		case RenderArea::Draw:
			drawAct->setChecked(true); 
			statusBar->showMessage (tr("Right click to close polygon"));
			break;

		case RenderArea::Cut:
			cutAct->setChecked(true); 
			statusBar->showMessage (tr("Right click to close polygon")); 
			break;
			default : break;
	}
	PaintInterf::updateToolBar(mode);
}

void DrawInterf::drawClicked() { updateToolBar(RenderArea::Draw); }
void DrawInterf::cutClicked() { updateToolBar(RenderArea::Cut); }

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//saisie d'un point

RenderAreaAppui::RenderAreaAppui(PaintInterfAppui& parent, const ParamMain& pMain, const QString& imageFile, const QPoint& PPrec) : DrawArea(parent,pMain,imageFile)
{
	Polygone P(RenderArea::Draw, false);
	if (PPrec!=QPoint(-1,-1)) P.modifQpolygon().push_back(PPrec);
	masque.push_back(P);
	Polygone P2(RenderArea::Draw, false);
	sauvegarde.push_back(P);
	display(imageFile);	//RenderArea::display(imageFile)
}

RenderAreaAppui::~RenderAreaAppui () {}

void RenderAreaAppui::mousePressEvent (QMouseEvent * event) { 
	if (event->button()!=Qt::LeftButton || (toolMode==RenderArea::Draw && masque.at(0).getQpolygon().count()==1)) return;
	DrawArea::mousePressEvent(event);
	if (toolMode==RenderArea::Draw) emit ptClicked();
}

void RenderAreaAppui::undoClicked() {
	QPoint P = getPoint();
	masque[0].modifQpolygon().clear();
	sauvegarde[0].modifQpolygon().clear();
	sauvegarde[0].modifQpolygon().push_back(P);
	update();
	parentWindow->updateToolBar(toolMode);
}
void RenderAreaAppui::redoClicked() {
	QPoint P = sauvegarde.at(0).getQpolygon().at(0);
	setPoint(P);
	parentWindow->updateToolBar(toolMode);
}

bool RenderAreaAppui::noPoint() const { return (masque.at(0).getQpolygon().count()==0); }
QPoint RenderAreaAppui::getPoint() const { return masque.at(0).getQpolygon().at(0); }
void RenderAreaAppui::setPoint(const QPoint& P) {
	sauvegarde[0].modifQpolygon().clear();
	masque[0].modifQpolygon().clear();
	if (P!=QPoint(-1,-1)) masque[0].modifQpolygon().push_back(P);
	update();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


PaintInterfAppui::PaintInterfAppui(const ParamMain* pMain, Assistant* help, const QList<QString>& pointsGPS, const QVector<QVector<QPoint> >& pointsAppui, QWidget* parent) : 
		DrawInterf(pMain,help,parent), ptsGPS(&pointsGPS), ptsAppui(pointsAppui)
{
	//RenderAreaAppui
	done = false;
	QString img = paramMain->getCorrespImgCalib().at(0).getImageTif();
	QPoint ptApp;	//on affiche par défaut le premier point d'appui de la liste
	int idx = getIndexPtApp(0);
	if (idx!=-1) ptApp = ptsAppui.at(0).at(idx);
	else ptApp = QPoint(-1,-1);
	renderArea.push_back( new RenderAreaAppui(*this,*paramMain, dir+paramMain->convertTifName2Couleur(img), ptApp) );
	if (!renderArea.at(0)->getDone())
		return;
	connect(renderArea[0], SIGNAL(ptClicked()), this, SLOT(ptClicked()));
	display();	//PaintInterf::display()

	createActions();
	createToolBar();
	toolBar->removeAction(cutAct);
	toolBar->removeAction(clearAct);
	drawAct->setEnabled(ptApp==QPoint(-1,-1));
	undoAct->setEnabled(ptApp!=QPoint(-1,-1));
	redoAct->setEnabled(false);
	connect(undoAct, SIGNAL(triggered()), this, SLOT(undoClicked()));
	connect(redoAct, SIGNAL(triggered()), this, SLOT(redoClicked()));

	//choix de l'image et du point GPS
	QLabel* label1 = new QLabel(tr("Image"));
	liste1 = new QComboBox;
	for (int i=0; i<paramMain->getCorrespImgCalib().count(); i++)
		liste1->addItem(paramMain->getCorrespImgCalib().at(i).getImageTif());
	QLabel* label2 = new QLabel(tr("GCP"));
	liste2 = new QComboBox;
	for (int i=0; i<ptsGPS->count(); i++)
		liste2->addItem(ptsGPS->at(i));
	liste1->setCurrentIndex(liste1->findText(img));
	liste2->setCurrentIndex(liste2->findText(ptsGPS->at(0)));

	QHBoxLayout *listesLayout = new QHBoxLayout;
	listesLayout->addWidget(label1);
	listesLayout->addWidget(liste1);
	listesLayout->addWidget(label2);
	listesLayout->addWidget(liste2);
	listesLayout->addStretch();

	QGroupBox* listesBox = new QGroupBox;
	listesBox->setLayout(listesLayout);;
	mainLayout->insertWidget(1,listesBox,0,Qt::AlignTop | Qt::AlignLeft);
	mainLayout->removeItem(mainLayout->itemAt(2));
	mainLayout->addStretch();

	connect(liste1, SIGNAL(currentIndexChanged(int)), this, SLOT(liste1Clicked()));
	connect(liste2, SIGNAL(currentIndexChanged(int)), this, SLOT(liste2Clicked()));

	setWindowTitle(conv(tr("GCP measure")));
	done = true;	
}

PaintInterfAppui::~PaintInterfAppui () {}

void PaintInterfAppui::liste1Clicked() {
	//modifie la visuArea à chaque changement de l'image
	renderArea.at(0)->display(dir+paramMain->convertTifName2Couleur(liste1->itemText(liste1->currentIndex())));
	liste2->setCurrentIndex(getIndexPtApp(liste1->currentIndex())); //modifie le point GPS correspondant
}

void PaintInterfAppui::liste2Clicked() {
	//modifie la visuArea à chaque changement de point GPS
	renderArea.at(0)->setPoint(ptsAppui.at(liste1->currentIndex()).at(liste2->currentIndex()));
	if (ptsAppui.at(liste1->currentIndex()).at(liste2->currentIndex())!=QPoint(-1,-1)) {
		drawAct->setEnabled(false);
		undoAct->setEnabled(true);
		redoAct->setEnabled(false);
	} else {
		drawAct->setEnabled(true);
		undoAct->setEnabled(false);
		redoAct->setEnabled(false);
	}
}

void PaintInterfAppui::undoClicked() {
	renderArea.at(0)->undoClicked();
	drawAct->setEnabled(true);
	undoAct->setEnabled(false);
	redoAct->setEnabled(true);
	//suppression du point dans la bd
	ptsAppui[liste1->currentIndex()][liste2->currentIndex()] = QPoint(-1,-1);
}
void PaintInterfAppui::redoClicked() {
	renderArea.at(0)->redoClicked();
	ptClicked();
}

void PaintInterfAppui::ptClicked() {
	drawAct->setEnabled(false);
	undoAct->setEnabled(true);
	redoAct->setEnabled(false);
	//ajout du point dans la bd
	QPoint P = renderArea.at(0)->getPoint();
	ptsAppui[liste1->currentIndex()][liste2->currentIndex()] = P;
}

void PaintInterfAppui::helpClicked() { assistant->showDocumentation(assistant->pageInterfAperoReference); } 

const QVector<QVector<QPoint> >& PaintInterfAppui::getPointsAppui() const { return ptsAppui; }

int PaintInterfAppui::getIndexPtApp(int img) const {
//index du premier point d'appu présent dans l'image courante
	int idx = 0;
	while (idx<ptsGPS->count() && ptsAppui.at(img).at(idx)==QPoint(-1,-1)) idx++;
	if (idx==ptsGPS->count()) idx = -1;
	return idx;
}

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//saisie d'un segment (2 render area)

RenderAreaSegment::RenderAreaSegment(PaintInterfSegment& parent, const QString& imageFile, const ParamMain& pMain, int N, int n, const QPoint& P1Prec, const QPoint& P2Prec) : DrawArea(parent,pMain,imageFile,N,n)
{
	maxPoint = (N==2) ? 1 : 2;
	Polygone P(RenderArea::Draw, false);
	if (P1Prec!=QPoint(-1,-1)) P.modifQpolygon().push_back(P1Prec);
	if (P2Prec!=QPoint(-1,-1)) P.modifQpolygon().push_back(P2Prec);
	masque.push_back(P);
	display(imageFile);	//RenderArea::display(imageFile)	
}

RenderAreaSegment::~RenderAreaSegment () {}

void RenderAreaSegment::mousePressEvent (QMouseEvent * event) { 
	if (event->button()==Qt::LeftButton && toolMode==RenderArea::Draw && masque.at(0).getQpolygon().count()==maxPoint) return;
	DrawArea::mousePressEvent(event);
	if (event->button()==Qt::LeftButton && toolMode==RenderArea::Draw)
		parentWindow->setLastPoint(npos);
	if (event->button()==Qt::LeftButton && toolMode==RenderArea::Draw && masque.at(0).getQpolygon().count()==maxPoint)
		emit sgtCompleted(true);
}

void RenderAreaSegment::paintEvent(QPaintEvent* event) {
	DrawArea::paintEvent(event);

	//dessin du segment
	if (masque.count()==0 || masque.at(0).getQpolygon().count()==0) return;
	if (maxPoint!=1) return; //cas de l'image unique pris en compte par DrawArea::paintEvent (segment et point)

	//récupération du point
	QPoint P1 = masque.at(0).getQpolygon().at(0);
	QPoint p1 = QPointF2QPoint(transfoInv(QPointF(P1)));	//coordonnées souris

	//affichage
	QPainter painter(this);	//=> points à dessiner en coordonnées souris
	painter.setRenderHint(QPainter::Antialiasing, true);
	QPen pen(QColor(255,0,0,255));
	pen.setWidth(1);
	painter.setPen(pen);

	//2 possiblités : 1 point (sur cette image) ou 2 points dans 2 images => 2 demi-segments
	//cas 1
	if (parentWindow->getNbPoint(npos)==0) {
		//painter.drawPoint(p1);	//on dessine plutôt une croix pour mieux le voir
		painter.drawLine(p1+QPoint(-10,-10),p1+QPoint(10,10));
		painter.drawLine(p1+QPoint(-10,10),p1+QPoint(10,-10));
	//cas 2
	} else if (parentWindow->getNbPoint(npos)>0) {
		//dessin du segment
		ParamDisplay posImg2 = parentWindow->getPosImg(npos); //récupération des paramètres de l'autre image
			//récupération du 2nd point
			QPoint P2 = (npos==1) ? parentWindow->getSegment().second : parentWindow->getSegment().first;
			//coordonnées souris
			QPointF p2bF = (QPointF(P2)-QPointF(posImg2.center))*posImg2.currentScale*posImg2.painterScale + QPointF(posImg2.size.width(),posImg2.size.height())/2.0;	//P2 dans l'image 2
			QPoint p2b(p2bF.x(),p2bF.y());
			QPoint p2 = QPointF2QPoint((P2-posImg2.center)*posImg2.currentScale*posImg2.painterScale + QPoint(posImg2.size.width(),posImg2.size.height())/2 + posImg2.origine);
			//point intermédiaire
			if (p1.x()==p2.x()) return;
			int x = (npos==2)? 0 : currentSize.width();
			QPoint p(x, p1.y()+double(p1.y()-p2.y())/double(p1.x()-p2.x())*(x-p1.x()));
			painter.drawLine(p1, p);
	}
	painter.end();
}

void RenderAreaSegment::undoClicked() {
	QPolygon* P = &(masque[0].modifQpolygon());
	if (sauvegarde.count()==0) {
		Polygone pol(RenderArea::Draw, false);
		sauvegarde.push_back(pol);
	}
	QPolygon* P2 = &(sauvegarde[0].modifQpolygon());
	P2->push_back(P->last());
	P->remove(P->count()-1);
	emit sgtCompleted(false);
	update();
	parentWindow->updateToolBar(toolMode);
}
void RenderAreaSegment::clearClicked() {
	QPolygon* P = &(masque[0].modifQpolygon());
	if (sauvegarde.count()==0) {
		Polygone pol(RenderArea::Draw, false);
		sauvegarde.push_back(pol);
	}
	QPolygon* P2 = &(sauvegarde[0].modifQpolygon());
	while (P->count()>0) {
		P2->push_back(P->last());
		P->remove(P->count()-1);
	}
	emit sgtCompleted(false);
	update();
	parentWindow->updateToolBar(toolMode);
}
void RenderAreaSegment::redoClicked() {
	QPolygon* P2 = &(sauvegarde[0].modifQpolygon());
	if (masque.count()==0) {
		Polygone pol(RenderArea::Draw, false);
		masque.push_back(pol);
	}
	QPolygon* P = &(masque[0].modifQpolygon());
	P->push_back(P2->last());
	P2->remove(P2->count()-1);
	if (P->count()==maxPoint) emit sgtCompleted(true);
	update();
	parentWindow->updateToolBar(toolMode);
}

bool RenderAreaSegment::noPolygone() const { return (masque.size()==0 || !(masque.at(0).getFerme())); }//cut->setEnabled()	refMasque==0
int RenderAreaSegment::getNbPoint() const { return (masque.count()>0)? masque.at(0).getQpolygon().count() : 0; }
bool RenderAreaSegment::ptSauvegarde() const { return (sauvegarde.count()>0)? (sauvegarde.at(0).getQpolygon().count()>0) : false; }
pair<QPoint,QPoint> RenderAreaSegment::getSegment() const { 
	const QPolygon* P = &(masque[0].getQpolygon());
	return pair<QPoint,QPoint>(P->at(0),P->at(1));
}
QPoint RenderAreaSegment::getPoint() const { return masque.at(0).getQpolygon().at(0); }

//////////////////////////////////////////////////////////////////////////////////////////////////////////


PaintInterfSegment::PaintInterfSegment(const ParamMain* pMain, Assistant* help, const pair<QString,QString>& images, QWidget* parent, bool planH, const QPoint& P1Prec, const QPoint& P2Prec) : 
		DrawInterf(pMain,help,parent), lastPoint(1)	//planH : pour l'axe des abscisses, sinon c'est l'échelle
{
	//RenderAreaSegment
	done = false;
	if (!planH)
		setWindowModality(Qt::NonModal);
	imageRef << images.first;
	if (images.second!=images.first)
		imageRef << images.second;	//images tif non tuilées

	int N = imageRef.count();
	for (int i=0; i<N; i++) {
		renderArea.push_back( new RenderAreaSegment(*this,imageRef.at(i),*paramMain,N,i+1,(N==2 && i==1)? P2Prec : P1Prec, (N==1)? P2Prec : QPoint(-1,-1)) );
		if (!renderArea.at(i)->getDone()) {
			return;
		}
		connect(renderArea[i], SIGNAL(sgtCompleted(bool)), this, SLOT(sgtCompleted(bool)));
	}
	display();	//PaintInterf::display()

	createActions();
	createToolBar();
	toolBar->removeAction(cutAct);
	if (planH) setWindowTitle(conv(tr("Abscissa axis selection")));
	else setWindowTitle(conv(tr("Scale selection")));
	done = true;	
}

PaintInterfSegment::~PaintInterfSegment () {}

void PaintInterfSegment::createActions() {
	DrawInterf::createActions();

	connect(undoAct, SIGNAL(triggered()), this, SLOT(undoClicked()));
	connect(redoAct, SIGNAL(triggered()), this, SLOT(redoClicked()));
	for (int i=0; i<renderArea.count(); i++) {
		connect(clearAct, SIGNAL(triggered()), renderArea[i], SLOT(clearClicked()));
	}
}

void PaintInterfSegment::undoClicked() {
	renderArea.at(lastPoint-1)->undoClicked();
	lastPoint = (imageRef.count()==1)? lastPoint : 3-lastPoint;	//pour le prochain undo
}
void PaintInterfSegment::redoClicked() {
	lastPoint = (imageRef.count()==1)? lastPoint : 3-lastPoint;	//pour le prochain redo
	renderArea.at(lastPoint-1)->redoClicked();
}

void PaintInterfSegment::sgtCompleted(bool completed) { drawAct->setEnabled(!completed); }

void PaintInterfSegment::helpClicked() { assistant->showDocumentation(assistant->pageDrawSegment); } 

pair<QPoint,QPoint> PaintInterfSegment::getSegment() const { 
	if (imageRef.count()==1) return renderArea.at(0)->getSegment();
	else return pair<QPoint,QPoint>(renderArea.at(0)->getPoint(), renderArea.at(1)->getPoint());
}
int PaintInterfSegment::getNbPoint(int n) const {
	int n2 = 2-n;
	return renderArea.at(n2)->getNbPoint();
}
bool PaintInterfSegment::ptSauvegarde(int n) const {
	int n2 = 2-n;
	return renderArea.at(n2)->ptSauvegarde();
}
void PaintInterfSegment::setLastPoint(int n) { lastPoint = n; }

//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//saisie d'un masque pour un plan


RenderAreaPlan::RenderAreaPlan(PaintInterfPlan& parent, const ParamMain& pMain, const QString& imageFile, const QString& masquePrec): 
	DrawArea( parent, pMain, imageFile ),
	maskPred(QString()),
	masqPrec( QImage() ),
	gradient( pair<QImage,QImage>( QImage(), QImage() ) ),
	regul( 0.1f ),
	autoRegul( true ),
	tempoImages( pair<QImage,QImage>( QImage(), QImage() ) ),
	refPainted( false )
{
	done = false;
	if (!loadMasquePrec (masquePrec)) {
		return;
	}
	withGradTool = true;
	display(imageFile);	//RenderArea::display(imageFile)
	done = true;
}

RenderAreaPlan::~RenderAreaPlan () {}

bool RenderAreaPlan::loadMasquePrec (const QString& masquePrec) {
	//récupère le masque de l'image masquePrec
	if (masquePrec.isEmpty()) {
		maskPred = QString();
		//tabRef.clear();
		return true;
	}
	if (!masqPrec.load(masquePrec.section(".",0,-2)+QString("nontuile.tif"))) {
			qMessageBox(parentWindow, tr("Read error"),conv(tr("Fail to read mask file.")));
			return false;
	}
	if (masqPrec.isNull()) {
			qMessageBox(parentWindow, tr("Read error"),tr("Empty image!"));
			return false;
	}
	maskPred = masquePrec;
	return true;
}

QSize RenderAreaPlan::sizeHint () {
	QSize s = DrawArea::sizeHint();
	tempoImages = pair<QImage,QImage>(QImage(),QImage());
	return s;
}

void RenderAreaPlan::updateGradRegul(int n) {
	int maxVal = parentWindow->maxSpinBox();
	if (n!=0) {
		//regul = 1.0 / double(maxVal - n + 1);
		regul = 1.0 / pow(10.0, maxVal - n);
		autoRegul = false;
	} else {
		//regul = currentScale/10.0;	//n=1/10 sans zoom, 1 avec zoom max
		regul = 0.1f;
		autoRegul = true;
	}
}

void RenderAreaPlan::mousePressEvent (QMouseEvent* event) { 
	DrawArea::mousePressEvent(event);
	if (event->button() == Qt::LeftButton) {
		switch (toolMode) {
			case RenderArea::Move:
				return;
			case RenderArea::ZoomIn:
			case RenderArea::ZoomOut:
				tempoImages = pair<QImage,QImage>(QImage(),QImage());
				return;
			default : break;
		}
		if (!withGradTool) return;

		//dessin du segment suivant les contours
		QPolygon* currentPolygone = &(masque[masque.count()-1].modifQpolygon());

		if (currentPolygone->size()==0) return;	// inutile en fait : soit >0, soit ==0 et le point a déjà été intégré par DrawArea::mousePressEvent donc ==1
		QPoint endPoint=QPointF2QPoint(transfo(event->pos()));
		QPoint lastPoint=currentPolygone->at(currentPolygone->count()-1);
		if (endPoint==lastPoint || (endPoint-lastPoint).manhattanLength()<=1) return;
		int rad = (1.0 / 2.0) + 2;

		QPolygon autoPath = findGradPath (lastPoint, endPoint);
		*currentPolygone += autoPath;
		undoCompteur.push_front(autoPath.size());
		rad += updateBuffer;
		update(transfoInv(QRect(lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad)));
	} else if (event->button() == Qt::RightButton)
		mouseDoubleClickEvent(event);
}

void RenderAreaPlan::mouseDoubleClickEvent (QMouseEvent* event) { 
	if ( (toolMode!=Draw) && (toolMode!=Cut) ) return;
	if (masque.size()==0) return;
	Polygone* currentPolygone = &masque[masque.count()-1];
	if (currentPolygone->getFerme()) return;	
	if (currentPolygone->getQpolygon().count()<2) return;	//il faut au moins 3 points avec celui-ci pour faire un polygone

	if (withGradTool) {
		QPolygon autoPath = findGradPath (currentPolygone->getQpolygon().at(currentPolygone->getQpolygon().count()-1), QPointF2QPoint(transfo(event->pos())));
		int n = 0;
		if (autoPath.count()!=0) {
			currentPolygone->modifQpolygon() += autoPath;	
			n = autoPath.size();
		}
		autoPath = findGradPath (currentPolygone->getQpolygon().at(currentPolygone->getQpolygon().count()-1), currentPolygone->getQpolygon().at(0));
		if (autoPath.count()!=0) {
			currentPolygone->modifQpolygon() += autoPath;	
			n += autoPath.size();
		}
		if (n!=0) undoCompteur.push_front(n);	
	} else {
		currentPolygone->modifQpolygon() += QPointF2QPoint(transfo(event->pos()));
		undoCompteur.push_front(1);	
	}
	currentPolygone->setFerme(true);

	update();
	if (toolMode==Draw && masque.size()==1) {	//if (toolMode==Draw && masque.size()==1 && maskPred.isEmpty())
		parentWindow->updateToolBar(toolMode);	//pour autoriser Cut
	}
}

void RenderAreaPlan::paintEvent(QPaintEvent* event) {
	if (!refPainted) {	//refPainted sert à ne dessiner le masque qu'une seule fois
		QPainter painter(&refImage);
			painter.setCompositionMode(QPainter::CompositionMode_Plus);
			painter.drawImage(0, 0, masqPrec);
		painter.end();
		masqPrec = QImage();
		refPainted = true;
	}
	DrawArea::paintEvent(event);
}

void RenderAreaPlan::calculeGradient (const qreal& scale, const QRegion& region) {
	//gradient de l'image restreinte à la zone region et réduite à 1/echelle (resultats dans gradient->first : gradient en x et gradient->second : gradient en y)
	if (region.isEmpty()) return;

	int filtreX [9] = {-1,0,1, -2,0,2, -1,0,1};
	int filtreY [9] = {-1,-2,-1, 0,0,0, 1,2,1};
	QSize size(refImage.size()*scale);
	gradient.first = QImage(size,refImage.format());
	gradient.second = QImage(size,refImage.format());
	gradient.first.fill(QColor(0,0,0,255).rgb());
	gradient.second.fill(QColor(0,0,0,255).rgb());
	QRect zone = region.boundingRect();
	zone.setSize(zone.size()*scale);
	zone.moveTo(zone.topLeft()*scale);
	QImage* img ;
	if (scale==painterScale/currentScale/2.0) {
		if (tempoImages.first.isNull())
			tempoImages.first = refImageClean.scaled(refImage.size()*scale);
		img = &tempoImages.first;
	} else {
		if (tempoImages.second.isNull())
			tempoImages.second = refImageClean.scaled(refImage.size()*scale);
		img = &tempoImages.second;
	}

	//calcul du gradient de refImage
	for (int x=zone.left(); x<zone.right()+1; x++) {
		if (x<=0 || x>=size.width()-1) continue;
		for (int y=zone.top(); y<zone.bottom()+1; y++) {
			if (y<=0 || y>=size.height()-1) continue;
			if (!(region.contains(QPoint(x,y)/scale))) continue;	//point hors zone
			float i=0;
			float j=0;
			for (int k=-1; k<2; k++) {
			for (int l=-1; l<2; l++) {
					if (k!=0)
						i += QColor(img->pixel(x-k,y-l)).value()*filtreX[(k+1)+(l+1)*3]/4.0;
					if (l!=0)
						j += QColor(img->pixel(x-k,y-l)).value()*filtreY[(k+1)+(l+1)*3]/4.0;
			}}
			gradient.first.setPixel(x,y,QColor(abs(int(i)),abs(int(i)),abs(int(i)),255).rgb());
			gradient.second.setPixel(x,y,QColor(abs(int(j)),abs(int(j)),abs(int(j)),255).rgb());
		}
	}
}

QPolygon RenderAreaPlan::plusCourtChemin (const QPoint& firstPoint, const QPoint& lastPoint, const QRegion& region, const qreal& scale, const QImage& distImg) {
	//algorithme du plus court chemin de firstPoint à lastPoint dans la zone region (sur l'image réduite à 1/echelle)
	if (region.isEmpty()) return 0;

	//nombre de points possibles
	int N = 0;
	QVector<QRect> V = region.rects();
	for (int i=0; i<V.count(); i++)
		N += V.at(i).width() * V.at(i).height();
	N *= scale;

	QPolygon polyg;
	QRect zone = region.boundingRect();
	zone.setSize(zone.size()*scale+QSize(1,1));
	zone.moveTo(zone.topLeft()*scale);
	QSize size(zone.size());
	
	//initialisation
	QVector<QVector<int> >coutTot(size.width(),QVector<int>(size.height(),numeric_limits<int>::max()));
	QPoint A = firstPoint*scale;
	QPoint B = lastPoint*scale;
	coutTot[A.x()-zone.x()][A.y()-zone.y()] = 0;
	QVector<QVector<QPoint> > prec(size.width(),QVector<QPoint>(size.height(),QPoint(-1,-1)));	//point précédent dans le chemin de firstPoint à lastPoint

	//paramètre de la droite (A,B) : ax+by+c=0 => contrainte chemin proche du segment
	float a = B.y()-A.y();
	float b = A.x()-B.x();
	float c = -b*A.y()-a*A.x();
	float n = 1.0/sqrt(a*a+b*b);

	/*int maxVal = 255;
	maxVal += (distImg.isNull()) ? int(regul*min(size.width(),size.height())*sqrt(2)) : 0;
	maxVal = maxVal * size.width()*size.height()+1;*/
	int maxVal = 255;
	maxVal += (distImg.isNull()) ? int(regul*min(size.width(),size.height())*sqrt(2.)) : 0;
	maxVal = maxVal * N/2;
	ElBornedIntegerHeap<QPoint,5> L(maxVal);	//par ordre croissant
     	L.push(A,0);
	while (L.nb()>0) {
		QList<QPoint> L2;
		while (L.nb()>0) {
			QPoint I;
			int Ci;
			L.pop(I, Ci);
			
			//les voisins de I :
			for (int i=-1; i<2; i++) {
			for (int j=-1; j<2; j++) {
				if (i==0 && j==0) continue; //point I
				QPoint J = I + QPoint(i,j);
				if (!(region.contains(J/scale))) continue;	//point hors zone

				float dist = (abs(J.x()*a+J.y()*b+c))*n;//-abs(I.x()*a+I.y()*b+c)

				int Cj = Ci + int(sqrt((pow(double(i*(255-QColor(gradient.first.pixel(J)).value())),2)+pow(double(j*(255-QColor(gradient.second.pixel(J)).value())),2))/(i*i+j*j)));	//on cherche le chemin avec le gradient maximum = la frontière
				Cj += (distImg.isNull()) ? dist*regul : QColor(distImg.pixel(J)).value();

				if (Cj<coutTot[J.x()-zone.x()][J.y()-zone.y()]) {
					coutTot[J.x()-zone.x()][J.y()-zone.y()] = Cj;
					prec[J.x()-zone.x()][J.y()-zone.y()] = I;	//point précédent J dans le plus court chemin
					if ( J!=B &&  Cj<coutTot[B.x()-zone.x()][B.y()-zone.y()] && !L2.contains(J)) {
						L2.push_back(J);
					}
				}
			}
			}
		}
		for (int i=0; i<L2.count(); i++) {
			L.push(L2.at(i), coutTot[L2.at(i).x()-zone.x()][L2.at(i).y()-zone.y()]);
		}
	}

	//récupération du chemin
	while (B!=A) {
		polyg.push_front(B/scale);
		B = prec[B.x()-zone.x()][B.y()-zone.y()];
	}
	
	return polyg;
}

QPolygon RenderAreaPlan::smoothPath (const QPoint& firstPoint, const QPolygon& polyg, int pas, float distance) {
	// suppression de pas-1 points sur pas s'ils sont suffisamment alignés avec leurs voisins
	QPolygon smoothPoly;
	int i = 0;

	//initialisation
	QPoint first = firstPoint;
	QPoint last = polyg.at(min(pas-1,polyg.size()-1));
		//paramètre de la droite (A,B) : ax+by+c=0
		float a = last.y()-first.y();
		float b = first.x()-last.x();
		float c = b*(first.x()-first.y());
		float n = 1.0/sqrt(a*a+b*b);
	while (i<polyg.size()-1) {
		QPoint courant = polyg.at(i);
		if (courant==last) {
			smoothPoly.push_back(courant);
			first = courant;
			last = polyg.at(min(i+pas,polyg.size()-1));
				a = last.y()-first.y();
				b = first.x()-last.x();
				c = -b*first.y()-a*first.x();
				n = 1.0/sqrt(a*a+b*b);
			i++;
			continue;
		}

		if (abs(courant.x()*a+courant.y()*b+c)*n>distance) {
			first = polyg.at(i);
			last = polyg.at(min(i+pas,polyg.size()-1));
				a = last.y()-first.y();
				b = first.x()-last.x();
				c = -b*first.y()-a*first.x();
				n = 1.0/sqrt(a*a+b*b);
			smoothPoly.push_back(courant);
		}
		i++;
	}

	smoothPoly.push_back(polyg.at(polyg.size()-1));
	return smoothPoly;
}

QPolygon RenderAreaPlan::findGradPath (const QPoint& firstPoint, const QPoint& lastPoint, const QImage& distImg) {
	if (firstPoint==lastPoint) return 0;

	//plus court chemin sur l'image réduite (échelle 1/2), dans le rectangle englobant (avec marges de gradientBuffer)
	qreal scale1 = min(painterScale/currentScale,1.0)/2.0;
	int gradientBuffer = 20.0/scale1;
	QRegion region( QRect(firstPoint,lastPoint).normalized().adjusted(-gradientBuffer, -gradientBuffer, +gradientBuffer, +gradientBuffer) );
	region &= refImage.rect();

	calculeGradient(scale1,region);
	QPolygon chemin = plusCourtChemin(firstPoint, lastPoint, region, scale1, distImg);
	chemin.push_front(firstPoint);

	//plus court chemin sur l'image taille réelle, dans la zone autour du chemin précédant (buffer de gradientBuffer2)
	qreal scale2 = min(painterScale/currentScale,1.0);
	int gradientBuffer2 = 10.0/scale2;
	region = QRegion();
	for (int i=0; i<chemin.size(); i++) {
		QPoint I =chemin.at(i);
		region += ( QRect(I.x()-gradientBuffer2, I.y()-gradientBuffer2, 2*gradientBuffer2, 2*gradientBuffer2) );	//pour des cercles : RegionType t = QRegion::Ellipse
	}
	region &= refImage.rect();

	calculeGradient(scale2,region);
	chemin = plusCourtChemin(firstPoint, lastPoint, region, scale2, distImg);

	return chemin;
}

Tiff_Im* RenderAreaPlan::endPolygone () {
	//enregistre le masque sous forme de QImage
	QApplication::setOverrideCursor( Qt::WaitCursor );

	//fermeture du polygone
	Polygone* lastPolygone = (masque.count()>0)? &(masque[masque.count()-1]) : 0;
	if (lastPolygone!=0) {
		lastPolygone->setFerme(true);
		update();
	}

	//initialisation de l'image
	QString tempofile = (maskPred.isEmpty())? applicationPath()+QString("/masquetempo.tif") : maskPred;	//fichier temporaire où est exporté le masque en attendant l'enregistrement
			//soit créé, soit maskPred est modifié
	if (maskPred.isEmpty() && QFile(tempofile).exists()) QFile(tempofile).remove();
	string fileMasq = tempofile.toStdString();

	Tiff_Im::SetDefTileFile(1<<20);
	Tiff_Im* imageMasq = 0;
	if (maskPred.isEmpty()) {	//tout en noir
		imageMasq = new Tiff_Im( fileMasq.c_str(), Pt2di( refImage.width(), refImage.height() ), GenIm::bits1_msbf, Tiff_Im::No_Compr, Tiff_Im::BlackIsZero );
		ELISE_COPY( imageMasq->all_pts(), 0, imageMasq->out() );	//fond noir
	} else {	//avec le masque précédent
		char buf[200];
		sprintf(buf,"%s", maskPred.toStdString().c_str() );
		ELISE_fp fp;
		if (! fp.ropen(buf,true)) {
			qMessageBox(this, tr("Read error"), conv(tr("Fail to read mask.")));
			QApplication::restoreOverrideCursor();
			return imageMasq;
		}
		fp.close();	
		imageMasq = new Tiff_Im(buf);
	}

	//remplissage de l'image par les polygones
	if (masque.count()>0) {
		for (int i=0; i<masque.size(); i++) {
			ElList<Pt2di> polyg;
			const QPolygon* p = &(masque.at(i).getQpolygon());
			for (int j=0; j<p->count(); j++) {
				polyg = polyg + Pt2di( p->at(j).x(), p->at(j).y() );
			}
			if (masque.at(i).getTmode()==Draw) {
				ELISE_COPY ( polygone(polyg), 1, imageMasq->out() );
			} else if (masque.at(i).getTmode()==Cut) {
				ELISE_COPY ( polygone(polyg), 0, imageMasq->out() );
			}
		}
	}

	//érosion : évite d'inclure les arrières-plans indésirables sur les bords du masque (! érode aussi l'image précédente)
	/*QImage erodeImg(tempofile);
	for (int i=0; i<erodeImg.width(); i++) {
	for (int j=0; j<erodeImg.height(); j++) {
		bool suppr = false;
		if (QColor(erodeImg.pixel(i,j)).value()==0) continue;
		for (int k=-1; k<2; k++) {
			for (int l=-1; l<2; l++) {
				//if (k!=0 && l!=0) continue;	//érosion à 4 voisinages
				if (i+k<0 || j+l<0 || i+k>erodeImg.width()-1 || j+l>erodeImg.height()-1) continue;
				if (QColor(erodeImg.pixel(i+k,j+l)).value()==0) {
					suppr = true;
					break;
				}
			}
			if (suppr) break;
		}
		if (!suppr) continue;
		ELISE_COPY ( rectangle(Pt2di(i-1,j-1),Pt2di(i+2,j+2)), 0, imageMasq->out() );		
	}}*/

	//copie du masque au format tif non tuilé
	/*QString tempofile2 = tempofile.section(".",0,-2)+QString("nontuile.tif");
	QString tempofile2blanc = tempofile2;
	QString commande = noBlank(applicationPath()) + QString("/lib/tiff2rgba ") + noBlank(tempofile) + QString(" ") + noBlank(tempofile2blanc);
	if (execute(commande)!=0) {
		qMessageBox(this, tr("Erreur d'exécution"), tr("Le masque enregistré n'a pas pas pu être converti au format tif non tuilé."));
		QApplication::restoreOverrideCursor();
		return imageMasq;
	}
		//transparence
	QImage img(tempofile2);
	QImage img2(img.size(), img.format());
	for (int x=0; x<img2.width(); x++) {
	for (int y=0; y<img2.height(); y++) {
		if (img.pixel(x,y)==QColor(0,0,0).rgb()) img2.setPixel(x,y,QColor(0,0,0,0).rgb());
		else img2.setPixel(x,y,QColor(0,255,0,125).rgb());
	}}
	img2.save(tempofile2);*/

	QApplication::restoreOverrideCursor();
	return imageMasq;
}

void RenderAreaPlan::continuePolygone (QList<Polygone>& conteneur, bool upDate) {
	DrawArea::continuePolygone(conteneur,upDate);
	if (conteneur.size()>0) {
	//si le polygone précédent était utilisé avec l'autre outil et n'a pas été fermé, on le supprime
		ToolMode otherTool = (toolMode==Draw) ? Cut : Draw;
		Polygone* polyg = &(conteneur[conteneur.count()-1]);
		//-- QRect rect = polyg->getQpolygon().boundingRect();
		ToolMode oldTmode = polyg->getTmode();
		if (oldTmode==otherTool && !(polyg->getFerme())) {
			polyg->modifQpolygon().clear();
			polyg->setTmode(toolMode);
			if (upDate)
				update();
		}
		else if (polyg->getFerme()) {
			Polygone newPolygone(toolMode, false);
			conteneur.push_back(newPolygone);
		}
	} else {
			Polygone newPolygone(toolMode, false);
			conteneur.push_back(newPolygone);
	}
}

void RenderAreaPlan::undoClicked() {
	int nb = undoCompteur.at(0);	//nombre de points du segment à supprimer
	int nb2=0;	//compteur des points que l'on supprime

	//polygone de masque à vider
	QPolygon* P = &masque[masque.count()-1].modifQpolygon();

	//création d'un nouveau polygone dans sauvegarde
	if ((sauvegarde.size()==0) || (masque[masque.count()-1].getFerme())) {
		sauvegarde.push_back(masque[masque.count()-1].clone());
	}
	//polygone de sauvegarde à remplir
	QPolygon* P2 = &(sauvegarde[sauvegarde.count()-1].modifQpolygon());

	while (nb2<nb) {
		P2->push_back(P->last());
		P->remove(P->count()-1);
		nb2++;
	}

	//mise à jour du masque
	masque[masque.count()-1].setFerme(false);	
	if (P->count()==0)	//pas de polygone vide dans masque
		masque.removeAt(masque.count()-1);

	//mise à jour de l'outil
	if (masque.size()>0) toolMode = masque[masque.count()-1].getTmode();	//le mode de dessin devient le mode du dernier point affiché, Draw sinon
	else toolMode = Draw;

	//mise à jour des compteurs
	redoCompteur.push_front(nb);
	undoCompteur.removeAt(0);	

	update();
	parentWindow->updateToolBar(toolMode);
}

void RenderAreaPlan::redoClicked() {
	int nb = redoCompteur.at(0);	//nombre de points du segment à rajouter
	int nb2=0;	//compteur des points que l'on ajoute

	//polygone de sauvegarde à vider
	Polygone* currentPolygone = &sauvegarde[sauvegarde.count()-1];
	QPolygon* P = &currentPolygone->modifQpolygon();

	//création d'un nouveau polygone dans masque
	if (masque.size()==0 || masque[masque.count()-1].getFerme()) {
		masque.push_back(Polygone(currentPolygone->getTmode(), false));
	}
	if (currentPolygone->getFerme() && nb==P->count()) masque[masque.count()-1].setFerme(true);	//le nouveau polygone sera fermé si on ajoute tout les points d'un polygone à fermer

	//polygone de masque à remplir
	QPolygon* P2 = &(masque[masque.count()-1].modifQpolygon());

	while (nb2<nb) {
		P2->push_back(P->last());
		P->remove(P->count()-1);
		nb2++;
	}

	//mise à jour de la sauvegarde	
	if (P->count()==0)	//pas de polygone vide dans la sauvegarde
		sauvegarde.removeAt(sauvegarde.count()-1);

	//mise à jour de l'outil
	toolMode = masque[masque.count()-1].getTmode();	//le mode de dessin devient le mode du dernier point affiché, Draw sinon

	//mise à jour des compteurs
	undoCompteur.push_front(nb);
	redoCompteur.removeAt(0);	

	update();
	parentWindow->updateToolBar(toolMode);
}

void RenderAreaPlan::clearClicked() {
	const QPolygon* P;
	QPolygon* P2;
	const Polygone* currentPolygone;
	for (int i=masque.count()-1; i>-1; i--) {
		//polygone de masque (à supprimer)
		currentPolygone = &(masque[i]);
		P = &(currentPolygone->getQpolygon());

		//création d'un nouveau polygone dans sauvegarde
		if ((sauvegarde.size()==0) || (currentPolygone->getFerme())) {
			sauvegarde.push_back(currentPolygone->clone());
		}
		P2 = &(sauvegarde[sauvegarde.count()-1].modifQpolygon());

		//remplissage
		for (int j=P->count()-1; j>-1; j--) {
			P2->push_back(P->at(j));
		}
	}

	masque.clear();
	update();
	toolMode = Draw;
	parentWindow->updateToolBar(toolMode);

	for (int i=0; i<undoCompteur.count(); i++) {
		redoCompteur.push_front(undoCompteur.at(i));
	}
	undoCompteur.clear();
}

void RenderAreaPlan::setGradTool (bool use) { withGradTool=use; }
bool RenderAreaPlan::noPolygone() const {return ((masque.size()==0 || !(masque.at(0).getFerme())) && masqPrec.isNull());}	//cut->setEnabled()	refMasque==0	 && tabRef.count()==0

//////////////////////////////////////////////////////////////////////////////////////////////////////////


PaintInterfPlan::PaintInterfPlan(const QString& imageFile, const ParamMain* pMain, Assistant* help, QWidget* parent, bool plan, const QString& masquePrec, bool filtre) : 
	DrawInterf( pMain, help, parent ),
	maskImg( 0 ),
	masqPrec( masquePrec ),
	filtrage( filtre )
	//plan=true si masque du plan horizontal, false si masque de corrélation=>définition renderArea à part
{
	if (plan) createRenderArea(imageFile);
}

PaintInterfPlan::~PaintInterfPlan () {
	delete maskImg;
	delete gradToolAct;
}

void PaintInterfPlan::createRenderArea(const QString& imageFile) {
	done = false;
	imageRef << imageFile;	//images tif non tuilées
	renderArea.push_back( new RenderAreaPlan(*this,*paramMain,imageFile,masqPrec) );
	if (!renderArea.at(0)->getDone()) {
		return;
	}

	display();	//PaintInterf::display()

	createActions();
	createToolBar();
	if (!filtrage) setWindowTitle(tr("Horizontal plan mask draw"));
	else setWindowTitle(tr("3D point mask modification"));
	done = true;
}

void PaintInterfPlan::createActions() {
	DrawInterf::createActions();

	for (int i=0; i<renderArea.count(); i++) {
		connect(undoAct, SIGNAL(triggered()), renderArea[i], SLOT(undoClicked()));
		connect(redoAct, SIGNAL(triggered()), renderArea[i], SLOT(redoClicked()));
		connect(clearAct, SIGNAL(triggered()), renderArea[i], SLOT(clearClicked()));
	}

	gradToolAct = new QAction(QIcon(g_iconDirectory+"paintsystem-icon.png"), conv(tr("Draw help")), this);
	gradToolAct->setCheckable (true);
	gradToolAct->setChecked (true);
	connect(gradToolAct, SIGNAL(triggered()), this, SLOT(gradToolClicked()));

	gradRegulBox = new QSpinBox;
	gradRegulBox->setEnabled (true);
	gradRegulBox->setMaximum(10);
	gradRegulBox->setMinimum(0);
	gradRegulBox->setSpecialValueText(tr("Auto"));
	//gradRegulBox->setSuffix(QString(" px"));
	gradRegulBox->setReadOnly(false);
	gradRegulBox->setValue(0);
	connect(gradRegulBox, SIGNAL(valueChanged(int)), this, SLOT(gradRegulChange(int)));
}

void PaintInterfPlan::createToolBar() {
	DrawInterf::createToolBar();
	toolBar->addSeparator ();
	toolBar->addAction(gradToolAct);
	toolBar->addWidget(gradRegulBox);
}

void PaintInterfPlan::gradToolClicked() {
	renderArea.at(0)->setGradTool(gradToolAct->isChecked());
	gradRegulBox->setEnabled(gradToolAct->isChecked());
}
void PaintInterfPlan::gradRegulChange(int i) { renderArea.at(0)->updateGradRegul(i); }

void PaintInterfPlan::helpClicked() {assistant->showDocumentation(assistant->pageDrawPlanCorrel); } 

void PaintInterfPlan::okClicked() {
	maskImg = renderArea.at(0)->endPolygone();
	DrawInterf::okClicked();
}

Tiff_Im* PaintInterfPlan::getMaskImg() {return maskImg;}
int PaintInterfPlan::maxSpinBox() const {return gradRegulBox->maximum();}
bool PaintInterfPlan::masqueIsEmpty() const { return renderArea.at(0)->masqueIsEmpty(); }


//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX//
//saisie d'un masque pour la corrélation (+ masque automatique)


RenderAreaCorrel::RenderAreaCorrel(PaintInterfCorrel& parent, const ParamMain& pMain, const QString& imageFile, const QString& masquePrec) : 
		RenderAreaPlan(parent,pMain,imageFile,masquePrec), autoMask(AutoMask()), ptsLiais(QList<pair<Pt2dr,QColor> >())
{}

RenderAreaCorrel::~RenderAreaCorrel () {}

void RenderAreaCorrel::autoClicked(int withHoles) {
//calcule un masque automatiquement à partir des points de liaisons présents dans l'image
	//initialisation
	QApplication::setOverrideCursor( Qt::WaitCursor );
	if (masque.count()>0 && !masque.at(masque.count()-1).getFerme()) {
		masque.removeAt(masque.count()-1);
		masque.removeAt(masque.count()-1);
	}
	
	clearClicked();	//dont update()

	//points du masque automatique
	if (autoMask.isNull()) {
		//lecture des points de liaison
		QList<Pt2dr> liaison_list;
		const QList<Pt2dr>* liaisons = &liaison_list;
		if (liaisons->count()==0) {
			qMessageBox(this, tr("Read error"), conv(tr("No tie-points found.")));
			QApplication::restoreOverrideCursor();
			return;
		}

		//récupération des paramètres de la caméra
		QVector<Pose> cameras(paramMain->getParamApero().getImgToOri().count());
		int N = -1;
		for (int i=0; i<paramMain->getParamApero().getImgToOri().count(); i++) {
			if (paramMain->convertTifName2Couleur(paramMain->getParamApero().getImgToOri().at(i))==fichierImage.section("/",-1,-1)) {
				N = i;
				break;
			}
		}
		if (N==-1) {
			qMessageBox(this, tr("Read error"), conv(tr("Fail to load image.")));
			QApplication::restoreOverrideCursor();
			return;
		}
		QString err = VueChantier::convert(paramMain, cameras, N);
		if (!err.isEmpty()) {
			qMessageBox(this, tr("Read error"), err);
			QApplication::restoreOverrideCursor();
			return;
		}

		const CamStenope* cam = &cameras.at(0).getCamera();
		//récupération des points 3D (a priori, ils sont dans le même ordre que les points 2D)
		QList<pair<Pt3dr, QColor> > listPt3D;
		QString err2 = VueChantier::getHomol3D (paramMain->getParamApero().getImgToOri().at(N), paramMain->getDossier(), listPt3D);
		if (!err2.isEmpty()) {
			qMessageBox(this, tr("Read error"), err2);
			QApplication::restoreOverrideCursor();
			return;
		}

		autoMask = AutoMask (*liaisons, listPt3D, refImageClean, cam, ptsLiais);
		//delete cam;
		
		if (!autoMask.isDone()) {
			qMessageBox(this, conv(tr("Execution error.")), conv(tr("Tie-point triangulation failed.")));
			QApplication::restoreOverrideCursor();
			return;
		}
	}
	const QList<QList<Pt2dr> >* boitesEnglob = &(autoMask.getBoitesEnglob());
	if (boitesEnglob->count()==0) {
		qMessageBox(this, tr("Read error"), conv(tr("No triangulation found")));
		QApplication::restoreOverrideCursor();
		return;
	}

	//dessin
	const QImage* distPtLiais = new QImage(autoMask.getDistPtLiais(currentSize));
	int n = 0;
	for (int i=0; i<boitesEnglob->count(); i++) {
		if (withHoles==0 && i>0) break;
		Polygone pol( (i==0)? Draw : Cut , true);
		for (int j=0; j<boitesEnglob->at(i).count(); j++) {
			QPoint P1( boitesEnglob->at(i).at(j).x , boitesEnglob->at(i).at(j).y );
			if (i==0) {
				int k = (j<boitesEnglob->at(i).count()-1)? j+1 : 0;
				QPoint P2( boitesEnglob->at(i).at(k).x , boitesEnglob->at(i).at(k).y );
				QPolygon pol2 = findGradPath(P1, P2, *distPtLiais);
				pol.modifQpolygon() += pol2;
				n += pol2.count();
			} else {
				pol.modifQpolygon() += P1;
				n++;
			}
		}
		masque.push_back(pol);
		Polygone pol3(pol);
		masque.push_back(pol3);
	}
	delete distPtLiais;
	undoCompteur.push_front(n);

	//mise à jour
	QApplication::restoreOverrideCursor();
	update();
	parentWindow->updateToolBar(toolMode);
}


void RenderAreaCorrel::paintEvent(QPaintEvent* event) {
	RenderAreaPlan::paintEvent(event);
	if (ptsLiais.count()!=0) {
		QPainter painterA(this);
		for (int i=0; i<ptsLiais.count(); i++) {
			QPen penA;
			penA.setColor(ptsLiais.at(i).second);
			penA.setWidth(7);
			painterA.setPen(penA);
			QPoint P(int(ptsLiais.at(i).first.x),int(ptsLiais.at(i).first.y));
		//	painterA.drawPoint(QPointF2QPoint(transfoInv(P)));	//on dessine plutôt une croix pour mieux le voir
			P = QPointF2QPoint(transfoInv(P));
			painterA.drawLine(P+QPoint(-10,-10),P+QPoint(10,10));
			painterA.drawLine(P+QPoint(-10,10),P+QPoint(10,-10));
		}
		painterA.end();
	}	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


PaintInterfCorrel::PaintInterfCorrel(const QString& imageFile, const ParamMain* pMain, Assistant* help, QWidget* parent, const QString& masquePrec) : 
		PaintInterfPlan(imageFile,pMain,help,parent,false,masquePrec), pointsLiaison(QList<Pt2dr>())
{
	done = false;
	//RenderAreaCorrel
	imageRef << imageFile;	//images tif non tuilées
	renderArea.push_back( new RenderAreaCorrel(*this,*paramMain,imageFile,masquePrec) );
	if (!renderArea.at(0)->getDone()) {
		return;
	}
	display();	//PaintInterf::display()

	createActions();
	createToolBar();
	setWindowTitle(tr("Mask draw"));
	done = true;
}

PaintInterfCorrel::~PaintInterfCorrel () {
	delete autoAct;
	delete autoFull;
	delete autoHoles;
}

void PaintInterfCorrel::createActions()
{
	PaintInterfPlan::createActions();
	autoAct = new QAction(QIcon(g_iconDirectory+"editbookmarks.png"), conv(tr("&Automatic mask")), this);
	QSignalMapper* mapper = new QSignalMapper(); 	
	connect(autoAct, SIGNAL(triggered()), this, SLOT(autoMenuDisplay()));
	autoFull = new QAction(conv(tr("create a full mask")), this);
	connect(autoFull, SIGNAL(triggered()), mapper, SLOT(map()));
	mapper->setMapping(autoFull, 0);
	autoHoles = new QAction(conv(tr("manage holes in mask")), this);
	connect(autoHoles, SIGNAL(triggered()), mapper, SLOT(map()));
	mapper->setMapping(autoHoles, 1);
	connect(mapper, SIGNAL(mapped(int)),renderArea.at(0), SLOT(autoClicked(int)));
}

void PaintInterfCorrel::createToolBar() {
	PaintInterfPlan::createToolBar();
	toolBar->addAction(autoAct);
}

void PaintInterfCorrel::autoMenuDisplay() {
	//affiche le menu de autoAct
	QMenu menu(toolBar);
	menu.addAction(autoFull);
	menu.addAction(autoHoles);
	menu.exec(toolBar->widgetForAction(autoAct)->mapToGlobal(QPoint(toolBar->widgetForAction(autoAct)->width(), 0)));	
}

const QList<Pt2dr>& PaintInterfCorrel::getPtsLiaison() {
	//nom de l'image tif
	QString imgCurrent = imageRef.at(0).section("/",-1,-1).section(".",0,-2);
	imgCurrent = imgCurrent.left(imgCurrent.count()-1);
	imgCurrent += QString(".tif");

	//lecture des points de liaison dans le dossier Pastis
	pointsLiaison.clear();
	cTplValGesInit<string>  aTpl;
	char** argv = new char*[1];
	char c_str[] = "rthsrth";
	argv[0] = new char[strlen( c_str )+1];
	strcpy( argv[0], c_str );
	cInterfChantierNameManipulateur* mICNM = cInterfChantierNameManipulateur::StdAlloc(1, argv, dir.toStdString(), aTpl );
	const vector<string>* aVN = mICNM->Get("Key-Set-HomolPastisBin");

		//fichier (*aVN)[aK]
	for (int aK=0; aK<signed(aVN->size()) ; aK++) {
		if (QString((*aVN)[aK].c_str()).contains("init")) continue;
		if (QString((*aVN)[aK].c_str()).contains("filtre")) continue;
	  	pair<string,string>  aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);
		if (imgCurrent.section("/",-1,-1)!=QString(aPair.first.c_str())) continue;

		//extraction des points d'appui
		ElPackHomologue aPack = ElPackHomologue::FromFile(dir.toStdString()+(*aVN)[aK]);
		if (aPack.size()==0) continue;
		for (ElPackHomologue::const_iterator  itH=aPack.begin(); itH!=aPack.end() ; itH++) {				
			//if (!liaisons->contains(itH->P1())) 
				pointsLiaison.push_back(itH->P1());	//pour qu'il y ait le même nb de points 3D que de points 2D
		}
	}
	delete [] argv[0];
	delete [] argv;
	delete mICNM;

	if (pointsLiaison.count()==0)
		qMessageBox(this, tr("Erreur de lecture"), conv(tr("No tie-points read.\nFail to compute mask automatically.")));
	return pointsLiaison;
}

void PaintInterfCorrel::helpClicked() { assistant->showDocumentation(assistant->pageDrawPlanCorrel); } 

//////////////////////////////////////////////////////////////////////////////////////////////////////////


Pt2dr TypeFPRIM::operator() (const pair<int,Pt2dr> & tq) const {return tq.second;}


BenchQdt::BenchQdt(TypeFPRIM Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN) :
              Box   (BOX), NbObj (NBOBJMAX), SZmin(SZMIN), qdt (Pt_of_Obj,Box,NBOBJMAX,SZMIN)  {}
         
bool BenchQdt::insert(const pair<int,Pt2dr> p,bool svp) {
             return qdt.insert(p,svp);
}

void BenchQdt::remove(const pair<int,Pt2dr>  & p) {
	qdt.remove(p);
}

void BenchQdt::clear() {
	qdt.clear();
}

void BenchQdt::voisins(Pt2dr p0,REAL ray, ElSTDNS set<pair<int,Pt2dr> >& S0) {
     qdt.RVoisins(S0,p0,ray);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


AutoMask::AutoMask() : boites(QList<QList<Pt2dr> >()), distPtLiais(QImage()), done(false), nullObject(true) {}
AutoMask::AutoMask(const AutoMask& autoMask) { copie(autoMask); }

AutoMask::AutoMask (const QList<Pt2dr>& liaisons, const QList<pair<Pt3dr, QColor> >& listPt3D, const QImage& refImage, const CamStenope* camera, QList<pair<Pt2dr,QColor> >& ptsLiais) : 
	boites(QList<QList<Pt2dr> >()), distPtLiais(QImage(refImage.size(), refImage.format())), done(false), nullObject(false)
{
//4 étapes :
//filtrage des points trop éloignés des autres pour ne conserver que ceux situés a priori sur l'objet principal :
	//tri des points par profondeur
	//conservation de la composante la plus proche de l'axe (on suppose que l'image est centrée sur l'objet, qu'il n'y a pas de masque au milieu, et que l'objet a suffisamment de points de liaisons sur sa partie la plus proche)
	//tri des points par voisinage
	//conservation de la plus grosse composante (on suppose que l'objet est en un seul morceau)
//boite englobante convexe des points de liaison conservés pour définir le masque (et calcul d'une triangulation de Delaunay de ces points)
//découpage dans cette boîte des triangles trop grands pour définir les trous et les concavités du masque :
	//recherche des triangles de côtés trop grands
	//tri des triangles par voisinage (trous différents)
	//définition d'un trou par la boîte englobant un lot de triangles contrainte par les triangles existants 
//affinage des contours par la méthode du plus court chemin (pondérée avec la distance aux points de liaison)
	done = false;
for (int i=0; i<liaisons.count(); i++)
ptsLiais.push_back( pair<Pt2dr,QColor>(liaisons.at(i),QColor(255,0,0)) );

	//tri des points par profondeur
		//profondeurs
	list<pair<double, int> > profondeur;
	for (int i=0; i<listPt3D.count(); i++)
		profondeur.push_back( pair<double, int>( abs(camera->ProfondeurDeChamps(listPt3D.at(i).first)) , i ) );
		//stat
	int profMax = 0;
	int profMin = numeric_limits<int>::max();
	for (list<pair<double, int> >::const_iterator it=profondeur.begin(); it!=profondeur.end(); it++) {
		if (it->first>profMax) profMax = it->first;
		if (it->first<profMin) profMin = it->first;
	}
	//double seuildisc = pow( double(profMax-profMin) * double(refImage.width()) * double(refImage.height()) / double(profondeur.size()) ,1.0/3.0);
		//tri par profondeur
	profondeur.sort();
	QList<pair<double, int> > profondeur2;	//conversion en QList
	for (list<pair<double, int> >::iterator it=profondeur.begin(); it!=profondeur.end(); it++)
		profondeur2.push_back(*it);
		//augmentation du nb de points proches en fct de la profondeur
	QList<pair<double,int> > sgmtProf;
	for (int i=0; i<profondeur2.count(); i++) {
		double a = double(i) / profondeur2.at(i).first * (profMax-profMin) / double(profondeur2.count());	//profondeur>0
		sgmtProf.push_back(pair<double, int>(a,profondeur2.at(i).second));
	}
		//minima locaux = changement de plan dominant => tri par composantes
	QList<int> profCompo;
	profCompo.push_back(0);
	double lastMax = sgmtProf.at(0).first;
	for (int i=0; i<sgmtProf.count(); i++) {
		if (i>0 && sgmtProf.at(i).first<sgmtProf.at(i-1).first && i<sgmtProf.count()-1 && sgmtProf.at(i).first<sgmtProf.at(i+1).first && sgmtProf.at(i).first<lastMax/2.0)
			profCompo.push_back(i);
		else if (i>0 && sgmtProf.at(i).first>sgmtProf.at(i-1).first && i<sgmtProf.count()-1 && sgmtProf.at(i).first>sgmtProf.at(i+1).first)
			lastMax = sgmtProf.at(i).first;
	}
	profCompo.push_back(sgmtProf.count());
		//distance moyenne des composantes à l'axe
	double minDistAxe = numeric_limits<int>::max();
	int bestCompo = -1;
	for (int i=0; i<profCompo.count()-1; i++) {
		if (profCompo.at(i+1)-profCompo.at(i)<4) continue;	//composante trop petite pour une triangulation
		double currentDist = 0;
		for (int j = profCompo.at(i); j<profCompo.at(i+1); j++) {
			Pt2dr P = liaisons.at( profondeur2.at(j).second );
			currentDist += (P.x-refImage.width()/2)*(P.x-refImage.width()/2) + (P.y-refImage.height()/2)*(P.y-refImage.height()/2);
		}
		currentDist /= double(profCompo.at(i+1)-profCompo.at(i));
		if (currentDist<minDistAxe) {
			minDistAxe = currentDist;
			bestCompo = i;
		}
	}
	if (bestCompo==-1) return;
		//plus proche composante
	QList<Pt2dr> liaisons0;
	for (int i=profCompo.at(bestCompo); i<profCompo.at(bestCompo+1); i++) {
		liaisons0.push_back( liaisons.at( profondeur2.at(i).second ) );
	}
for (int i=0; i<liaisons0.count(); i++)
ptsLiais.push_back( pair<Pt2dr,QColor>(liaisons0.at(i),QColor(0,255,0)) );

	//tri des points 2D
		//critère
	REAL maxLgr = max(refImage.width()*refImage.width()/liaisons0.count(), refImage.height()*refImage.height()/liaisons0.count());
	REAL distMax = sqrt(maxLgr);
		//recherche des voisins
	TypeFPRIM Pt_of_Point;
	Box2dr box(Pt2dr(0,0), Pt2dr(refImage.width(),refImage.height()) );
	BenchQdt bench(Pt_of_Point, box, liaisons0.count(), 1.0);
	for (int i=0; i<liaisons0.count(); i++) {
		bench.insert(pair<int,Pt2dr>(i,liaisons0.at(i)));
	}
	QVector<ElSTDNS set<pair<int,Pt2dr> > > vois(liaisons0.count());
	for (int i=0; i<liaisons0.count(); i++) {
		bench.voisins(liaisons.at(i), distMax, vois[i]);
		pair<int,Pt2dr> A(i,liaisons0.at(i));
	};

		//tri en composantes connexes
	QList<QList<int> > compo;
	for (int i=0; i<liaisons0.count(); i++) {
		//on vérifie que ce point n'est pas déjà dans une composante
		QList<int>* p = 0;
		bool b = false;
		if (compo.count()>0) {
			for (int j=0; j<compo.count(); j++) {
				if (compo.at(j).contains(i)) {
					if (b) {
						for (int k=0; k<compo.at(j).count(); k++) {
							if (!p->contains(compo.at(j).at(k)))
								p->push_back(compo.at(j).at(k));
						}
						compo[j].clear();
					} else {
						b = true;
						p = &(compo[j]);
					}
				}
			}
		}
		if (!b) {
			compo.push_back(QList<int>());
			p = &(compo[compo.count()-1]);
			p->push_back(i);
		}
		if (vois[i].size()>0) {
			for (ElSTDNS set<pair<int,Pt2dr> >::const_iterator it=vois[i].begin(); it!=vois[i].end(); it++) {
				if (!p->contains(it->first))
					p->push_back(it->first);
			}	
		}
	}
		//recherche de la plus grande composante
	int nbMax = 0;
	int numCompo = -1;
	for (int i=0; i<signed(compo.count()); i++) {
		if (compo.at(i).size()>nbMax) {
			nbMax = compo.at(i).size();
			numCompo = i;
		}
	}
		//récupération des points
	QList<Pt2dr> liaisons2;
	for (int i=0; i<compo.at(numCompo).count(); i++) {
		liaisons2.push_back( liaisons0.at( compo.at(numCompo).at(i) ) );
	}
	compo.clear();
/*for (int i=0; i<liaisons2->count(); i++)
ptsLiais.push_back( pair<Pt2dr,QColor>(liaisons2->at(i),QColor(0,0,255)) );*/

	//triangulation et boîte englobante
	QList<Pt2dr> boite;
	struct triangulateio triangulation;
	if (!boiteEnglob (liaisons2, boite, triangulation)) return;
	int numberoftriangles = triangulation.numberoftriangles;
	if (numberoftriangles==0) return;
	double aireTot = aireBoite(boite);

	//récupération des triangles trop grands (trous dans la triangulation)
		//triangles
	QList<QVector<int> > grandTriangles;	//int = index des points dans liaisons2
	QList<double> aires;
	for (int i=0; i<numberoftriangles; i++) {
		QVector<int> triangle;
		QList<Pt2dr> triangle2;
		for (int j=0; j<3; j++) {
			triangle.push_back( triangulation.trianglelist[3*i+j] );
			triangle2.push_back( liaisons2.at( triangle.at(j) ) );			
		}
		//vérification du critère
		if (!testTriangles(triangle2, 2*maxLgr)) {
//if (!testTriangles(triangle2, maxLgr)) {
			grandTriangles.push_back(triangle);
			aires.push_back(aireTriangle(triangle2));
		}
	}
		//libération de la mémoire
	freeTriangulateio(triangulation);

	//tri des triangles retenus par composante connexe
	QList<QList<int> > composantes;
	QList<double> airesCompo;
	if (grandTriangles.count()>0) {
		for (int i=0; i<grandTriangles.count(); i++) {
			//init
			if (composantes.count()==0)  {
				QList<int> l;
				l.push_back(i);
				composantes.push_back(l);
				airesCompo.push_back(aires.at(i));
				continue;
			}
			//recherche des voisins
			bool b =false;
			int n = -1;
			for (int j=0; j<composantes.count(); j++) {
				for (int k=0; k<composantes.at(j).count(); k++) {
					if (compareTriangles(grandTriangles.at(i), grandTriangles.at(composantes.at(j).at(k)))) {
						if (!b) {
							b = true;
							composantes[j].push_back(i);
							airesCompo[j] += aires.at(i);
							n = j;
						} else {
							//on relie les 2 composantes
							composantes[n].append(composantes.at(j));
							airesCompo[n] += airesCompo.at(j);
							composantes.removeAt(j);
							airesCompo.removeAt(j);
							j--;
						}
						break;
					}
				}
				//on vérifie que ce triangle n'est pas voisin d'une autre composante
			}
			if (b) continue;
			//pas de voisin répertorié => nouvelle composante
			QList<int> l;
			l.push_back(i);
			composantes.push_back(l);
			airesCompo.push_back(aires.at(i));
		}
	}

	//tri des composantes par nombre de triangles
	boites.push_back(boite);
	if (composantes.count()>0) {
		for (int i=0; i<composantes.count(); i++) {
			//if (composantes.at(i).count()>numberoftriangles/100) continue;
			if (airesCompo.at(i)>aireTot/100) continue;
			composantes.removeAt(i);
			airesCompo.removeAt(i);
			i--;
		}
	}
/*
	//extraction de la boite englobante (pas convexe)
	if (composantes.count()>0) {
		for (int i=0; i<composantes.count(); i++) {	//composante i
	cout << "nb pt compo " << composantes.at(i).count() << "\n";
			QList<pair<int,int> > frontiere;
			QList<pair<int,int> > liste; //liste des segments n'appartenant pas à la limite
			for (int j=0; j<composantes.at(i).count(); j++) {	//triangle j
				for (int k=0; k<3; k++) {	//sommet k
					//on vérifie que le segment est sur la limite (il n'appartient qu'à un seul triangle)
					int point1 = grandTriangles->at(composantes.at(i).at(j)).at(k);
					int l = (k==2) ? 0 : k+1;
					int point2 = grandTriangles->at(composantes.at(i).at(j)).at(l);
					if (liste.contains(pair<int,int>(point1,point2))) continue;
					if (liste.contains(pair<int,int>(point2,point1))) continue;

					bool b = true;
					for (int j2=0; j2<composantes.at(i).count(); j2++) {	//triangle j2	
						if (j2==j) continue;
						for (int k2=0; k2<3; k2++) {	//sommet k2
							int point12 = grandTriangles->at(composantes.at(i).at(j2)).at(k2);
							int l2 = (k2==2) ? 0 : k2+1;
							int point22 = grandTriangles->at(composantes.at(i).at(j2)).at(l2);
							if ( pair<int,int>(point1,point2)==pair<int,int>(point12,point22) || pair<int,int>(point1,point2)==pair<int,int>(point22,point12) )		{
								b = false;
								liste.push_back(pair<int,int>(point1,point2));
								break;
					   		}
						}	
						if (!b) break;
					}
					if (b) frontiere.push_back(pair<int,int>(point1,point2));		
				}
			}
	cout << "frontiere\n";
	cout << "nb pt front " << frontiere.count() << "\n";
			int beginPoint = frontiere.at(0).first;
			QList<Pt2dr>* boite2 = new QList<Pt2dr>;
			boite2->push_back(liaisons2->at(beginPoint));
	cout << "beginPoint\n";
			int point1 = beginPoint;
			int point2 = frontiere.at(0).second;
			while (point2!=beginPoint) {
				boite2->push_back(liaisons2->at(point2));
				for (int i=0; i<frontiere.count(); i++) {
					if (frontiere.at(i).first==point2) {
						point2 = frontiere.at(i).second;
						point1 = frontiere.at(i).first;
						break;
					} else if (frontiere.at(i).second==point2 && frontiere.at(i).first!=point1) {
						point2 = frontiere.at(i).first;
						point1 = frontiere.at(i).second;
						break;
					}
				}
			}
			boites.push_back(boite2);
	cout << "boite2\n";
		}
	}
*/
	for (int i=0; i<composantes.count(); i++) {
		for (int j=0; j<composantes.at(i).count(); j++) {
			QList<Pt2dr> triangle2;
			for (int k=0; k<3; k++) {
				triangle2.push_back( liaisons2.at( grandTriangles.at( composantes.at(i).at(j) ).at(k) ) );			
			}
			boites.push_back(triangle2);
		}
	}


/*
	//boîte englobante de chaque composante trouvée
	boites = new QList<QList<Pt2dr>*>;
	boites.push_back(boite);
getchar();
	for (int i=0; i<composantes.count(); i++) {
cout << "triangulation " << (i+1) << "\n";
		if (composantes.at(i).count()<numberoftriangles/100) continue;
		//points de la composante
		QList<Pt2dr>* liaisons3 = new QList<Pt2dr>;
		for (int j=0; j<composantes.at(i).count(); j++) {
			for (int k=0; k<3; k++) {
				Pt2dr P = liaisons2->at( grandTriangles->at( composantes.at(i).at(j) ).at(k) );
				if (!liaisons3->contains(P)) liaisons3->push_back(P);
			}
		}
		//boite englobante
		QList<Pt2dr>* boite2 = new QList<Pt2dr>;
		struct triangulateio* triangulation2 = new struct triangulateio;
		boiteEnglob (liaisons3, boite2, triangulation2);
		boites.push_back(boite2);

		//libération de la mémoire
		delete liaisons3;
		freeTriangulateio(triangulation2);
	}*/

	//image de la disance au plus proche point
	for (int j=0; j<refImage.width(); j++) {
	for (int k=0; k<refImage.height(); k++) {
		distPtLiais.setPixel(j,k, QColor(int(255),int(255),int(255),255).rgb() );
	}}
	for (int i=0; i<liaisons2.count(); i++) {
		for (int j=-distMax; j<distMax; j++) {
		for (int k=-distMax; k<distMax; k++) {
			QPoint P(liaisons2.at(i).x+j,liaisons2.at(i).y+k);
			if (P.x()<0 || P.y()<0 || P.x()>refImage.width()-1 || P.y()>refImage.height()-1) continue;
			double px = min( distMax,sqrt( (double)(j*j+k*k) ) )*255.0/distMax;
			//px = floor((distMax-px)*255.0/distMax);
			distPtLiais.setPixel(P.x(),P.y(), QColor(int(px),int(px),int(px),255).rgb() );
		}}
	}

//autre algorithme : http://www.iag.asso.fr/articles/nuage.htm
	done = true;
}

AutoMask::~AutoMask () {}

bool AutoMask::boiteEnglob (const QList<Pt2dr>& points, QList<Pt2dr>& boite, struct triangulateio& triangulation) const {
	//écriture des points au format poly
	if (points.count()<3) return false;
	struct triangulateio in;
	in.numberofpoints = points.count();
	in.numberofpointattributes = 0;
	in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
	in.numberofsegments = 0;
	in.numberofholes = 0;
	in.numberofregions = 0;
	for (int i=0; i<points.count(); i++) {
		in.pointlist[2*i] = points.at(i).x;
		in.pointlist[2*i+1] = points.at(i).y;
	}

	//initialisation des sorties
	triangulation.pointlist = (REAL *) NULL; 
	triangulation.pointmarkerlist = (int *) NULL;
	triangulation.trianglelist = (int *) NULL;
	triangulation.segmentlist = (int *) NULL;
	triangulation.segmentmarkerlist = (int *) NULL;

	// Triangulation : convex hull (c), numérotation à partir de zéro (z)
	triangulate("cz", &in, &triangulation, (struct triangulateio *) NULL);	//http://www.cs.cmu.edu/~quake/triangle.html
	//if (triangulation.segmentlist==NULL) return false;

	//récupération de la boîte englobante
	for (int i=0; i<triangulation.numberofsegments; i++)
		boite.push_back( points.at(triangulation.segmentlist[i*2]) );

	//libération de la mémoire
	freeTriangulateio(in);
	return true;
}

void AutoMask::freeTriangulateio(struct triangulateio& triangulation) const {
	//libération de la mémoire
	free(triangulation.pointlist);
	free(triangulation.pointattributelist);
	free(triangulation.pointmarkerlist);
	free(triangulation.trianglelist);
	free(triangulation.triangleattributelist);
	free(triangulation.trianglearealist);
	free(triangulation.neighborlist);
	free(triangulation.segmentlist);
	free(triangulation.segmentmarkerlist);
	free(triangulation.edgelist);
	free(triangulation.edgemarkerlist);
}

bool AutoMask::compareTriangles(const QVector<int>& triangle1, const QVector<int>& triangle2) const {
	//vérifie si les triangles sont voisins (arête commune)
	bool a = false;
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			if (triangle1.at(i)!=triangle2.at(j)) continue;
			if (!a) {	//1ier point commun
				a = true;
				break;
			}
			//2nd point commun
			return true;
		}
	}
	return false;	//0 ou 1 point commun
}

bool AutoMask::testTriangles(const QList<Pt2dr>& triangle, double critere) const {
	//vérifie si le triangle a au moins 2 côtés trop grands
	bool a = false;
	for (int i=0; i<3; i++) {
		int j = (i==2)? 0 : i+1;
		double dist = (triangle.at(i).x-triangle.at(j).x) * (triangle.at(i).x-triangle.at(j).x) + (triangle.at(i).y-triangle.at(j).y) * (triangle.at(i).y-triangle.at(j).y);
		if (dist<critere) continue;
		if (!a) a = true;
		else return false;
	}
	return true;	//triangle petit
}

double AutoMask::aireTriangle(const QList<Pt2dr>& triangle) const {
	//calcule l'aire d'un triangle <AB;AC>/2
	double a = (triangle.at(1).x-triangle.at(0).x)*(triangle.at(2).x-triangle.at(0).x) + (triangle.at(1).y-triangle.at(0).y)*(triangle.at(2).y-triangle.at(0).y);
	return abs(a)/2.0;	
}

double AutoMask::aireBoite(const QList<Pt2dr>& boite) const {
	//calcule l'aire d'une boite englobante
	double a = 0;
	for (int i=0; i<boite.count(); i++) {
		int j = (i==boite.count()-1) ? 0 : i+1;
		a += (boite.at(j).x-boite.at(i).x) * (boite.at(j).y+boite.at(i).y)/2.0;
	}
	return abs(a);	
}

QImage AutoMask::getDistPtLiais(const QSize& size) const {
	QImage img(size, distPtLiais.format());
	for (int i=0; i<floor((double)size.width()); i++) {
	for (int j=0; j<floor((double)size.height()); j++) {
		img.setPixel(i, j, distPtLiais.pixel(i,j) );
	}}
	return img;
}

AutoMask& AutoMask::operator=(const AutoMask& autoMask) {
	if (this!=&autoMask) copie(autoMask);
	return *this;
}

void AutoMask::copie(const AutoMask& autoMask) {
	boites = autoMask.getBoitesEnglob();
	distPtLiais = autoMask.getDistPtLiais();
	done = autoMask.isDone();
	nullObject = autoMask.isNull();
}

const QList<QList<Pt2dr> >& AutoMask::getBoitesEnglob() const { return boites; }
const QImage& AutoMask::getDistPtLiais() const { return distPtLiais; }
bool AutoMask::isDone() const { return done; }
bool AutoMask::isNull() const { return nullObject; }
