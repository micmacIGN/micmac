#if defined Q_WS_WIN 
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "vueChantier.h"

#define GLWidget GLWidget_IC

using namespace std;


void multTransposeMatrix(const GLdouble* m) {	//pour Windows
	GLdouble* m2 = new GLdouble[16];
	for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
		m2[4*i+j] = m[4*j+i];
	glMultMatrixd(m2);
	delete m2;
}


//initialisation/////////////////////////////////////////////////
double GLWidget::visibilite = 100.0;//1000.0;
GLWidget::GLWidget(VueChantier *parent, GLParams* params, const ParamMain& pMain) :
	QGLWidget(parent),
	parametres(params),
	espace(QVector<GLdouble>(6)),
	currentDezoom(QVector<int>()),
	paramMain(&pMain),
	objectCam(0),
	objectEmp(0),
	objectApp(0),
	objectNuag(QVector<GLuint>()),
	axes(0),
	info(false),
	ref(false)
{     
	setFocusPolicy(Qt::WheelFocus);
	setAutoFillBackground(false);

	infoBulle.second = -1;

	//espace représenté
        GLdouble longueurChantier = max(max(parametres->getZoneChantierEtCam().at(1)-parametres->getZoneChantierEtCam().at(0),parametres->getZoneChantierEtCam().at(3)-parametres->getZoneChantierEtCam().at(2)),parametres->getZoneChantierEtCam().at(5)-parametres->getZoneChantierEtCam().at(4));
	espace[4] = 1;
	espace[5] = visibilite*longueurChantier;
        GLdouble em = longueurChantier/(espace.at(5)+espace.at(4));
	espace[0] = -em;	//pour que le chantier prenne toute la fenêtre à une profondeur moyenne et une focale moyenne
	espace[1] = em;
	espace[2] = -em;
	espace[3] = em;

	//sélection de caméras
	addCamAct = new QAction(conv(tr("Select this camera")), this);
	connect(addCamAct, SIGNAL(triggered()), this, SLOT(addCam()));
}
GLWidget::~GLWidget() {  
	makeCurrent();    
/*	glDeleteLists(objectCam, 1);   
	glDeleteLists(objectApp, 1);  
	glDeleteLists(axes, 1);  
	glDeleteLists(boule, 1);   
	if (parametres->getNuages().count()>0) {  
		for (int i=0; i<6*parametres->getNuages().count(); i++)
			glDeleteLists(objectNuag[i], 1);
	} */
	for (GLuint i=1; i<=parametres->getNbGLLists(); i++)
		glDeleteLists(i, 1);  
	parametres->resetNbGLLists();
}
QSize GLWidget::sizeHint() const {
	return QSize(400, 400);
}

void GLWidget::initializeGL() {
	qglClearColor(QColor(0,0,50));

        if (parametres->getRot()==0) {
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
                parametres->modifRot() = new GLdouble[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, parametres->modifRot());
	}
        parametres->modifTrans()[0] = -(parametres->getZoneChantierEtCam().at(0)+parametres->getZoneChantierEtCam().at(1))/2.0;
        parametres->modifTrans()[1] = -(parametres->getZoneChantierEtCam().at(2)+parametres->getZoneChantierEtCam().at(3))/2.0;
        parametres->modifTrans()[2] = -(parametres->getZoneChantierEtCam().at(4)+parametres->getZoneChantierEtCam().at(5))/2.0;

	if (parametres->getNuages().count()>0) {	
                currentDezoom.resize(parametres->getNuages().count());
                currentDezoom.fill(-1);
                for (int i=0; i<parametres->getNuages().count(); i++)
			calcDezoom(i);
	}
	for (GLuint i=1; i<=parametres->getNbGLLists(); i++)
		glDeleteLists(i, 1); 
	parametres->resetNbGLLists();
	objectCam = makeObjectCam();
	objectEmp = makeObjectEmp();
	objectApp = makeObjectApp();
	axes = makeAxes();
	boule = makeBoule();
	if (parametres->getNuages().count()>0) {
		objectNuag.resize(6*parametres->getNuages().count());
		for (int i=0; i<parametres->getNuages().count(); i++) {
			for (int j=0; j<6; j++)
				objectNuag[6*i+j] = makeObjectNuag(i,j);
		}
		 /*doNuages();*/
	}
	glShadeModel(GL_FLAT);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
        int f = parametres->getFocale();
        parametres->setFocale(0);
	modifFocale(f);	//ne modifie rien si focale = le paramètre
	//setupViewport(width(), height());
}

void GLWidget::doNuages() {
	//threads
	if (parametres->getNuaglayers().count()==0 || !parametres->getNuaglayers().contains(true)) return;
	/*QVector<NuageThread*> nuageThread(6*parametres->getNuages().count(),0);
        for (int i=0; i<parametres->getNuages().count(); i++) {
		for (int j=0; j<6; j++)
			nuageThread[6*i+j] = new NuageThread(this,i,j);
	}
	for (int i=0; i<6*parametres->getNuages().count(); i++)
		nuageThread[i]->start();
	for (int i=0; i<6*parametres->getNuages().count(); i++)
		while (nuageThread[i]->isRunning()) {}
	for (int i=0; i<6*parametres->getNuages().count(); i++) {
		objectNuag[i] = nuageThread[i]->getObjectNuag();
	}
	for (int i=1; i<6*parametres->getNuages().count(); i++)
		delete nuageThread[i];*/

        for (int i=0; i<parametres->getNuages().count(); i++)
		for (int j=0; j<6; j++)
			if (objectNuag[6*i+j]!=0) glDeleteLists(objectNuag[6*i+j], 1); 
        for (int i=0; i<parametres->getNuages().count(); i++)
		for (int j=0; j<6; j++)
			objectNuag[6*i+j] = makeObjectNuag(i,j);
}


GLuint GLWidget::makeObjectCam() {
//dessin des emprises et des caméras
	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);	//vue 3D des objets : rotations et translation

	//choix des couleurs
	QList<QColor> couleurs;
	for (int i=0; i<parametres->getPoses().count(); i++) {
		int x = 255*i/(parametres->getPoses().count()-1);
		QColor c(255-x,x,(pow(double(-1),i)+1)*255/2);
		couleurs.push_back(c);
	}

	//caméras
	glLineWidth(3);	
	glPointSize(7);	
	for (int i=0; i<parametres->getPoses().count(); i++) {
		if (info && i==infoBulle.second)
			qglColor(QColor(255,0,0));
		else if (ref && i==infoBulle.second)
			qglColor(QColor(255,255,0));
		else
			qglColor(couleurs.at(i));
		REAL f = parametres->getPoses().at(i).getCamera().Focale();
		Pt3dr C = parametres->getPoses().at(i).centre();
                Pt3dr P1 = parametres->getPoses().at(i).getCamera().ImEtProf2Terrain(Pt2dr(0,0),parametres->getDistance()*f);
                Pt3dr P2 = parametres->getPoses().at(i).getCamera().ImEtProf2Terrain(Pt2dr(parametres->getPoses().at(i).width(),0),parametres->getDistance()*f);
                Pt3dr P3 = parametres->getPoses().at(i).getCamera().ImEtProf2Terrain(Pt2dr(0,parametres->getPoses().at(i).height()),parametres->getDistance()*f);
                Pt3dr P4 = parametres->getPoses().at(i).getCamera().ImEtProf2Terrain(Pt2dr(parametres->getPoses().at(i).width(),parametres->getPoses().at(i).height()),parametres->getDistance()*f);
		glBegin(GL_TRIANGLES);
			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P1.x, P1.y, P1.z);
			glVertex3d(P2.x, P2.y, P2.z);

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P1.x, P1.y, P1.z);
			glVertex3d(P3.x, P3.y, P3.z);

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P4.x, P4.y, P4.z);
			glVertex3d(P2.x, P2.y, P2.z);

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P4.x, P4.y, P4.z);
			glVertex3d(P3.x, P3.y, P3.z);			
		glEnd();
		glBegin(GL_QUADS);
			glVertex3d(P1.x, P1.y, P1.z);
			glVertex3d(P2.x, P2.y, P2.z);
			glVertex3d(P4.x, P4.y, P4.z);
			glVertex3d(P3.x, P3.y, P3.z);				
		glEnd();
		//point quand on est trop éloigné pour voir la cméra
		glBegin(GL_POINTS);
			glVertex3d(C.x, C.y, C.z);
		glEnd();
		glBegin(GL_LINES);
			qglColor(QColor(0,0,0));
			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P1.x, P1.y, P1.z);

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P2.x, P2.y, P2.z);

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P3.x, P3.y, P3.z);	

			glVertex3d(C.x, C.y, C.z);
			glVertex3d(P4.x, P4.y, P4.z);			
		glEnd();
	}

	glEndList();
	return list;
}

GLuint GLWidget::makeObjectEmp() {
//dessin des emprises et des caméras
	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);	//vue 3D des objets : rotations et translation

	//choix des couleurs
	QList<QColor> couleurs;
	for (int i=0; i<parametres->getPoses().count(); i++) {
		int x = 255*i/(parametres->getPoses().count()-1);
		QColor c(255-x,x,(pow(double(-1),i)+1)*255/2);
		couleurs.push_back(c);
	}

	//emprise
	glLineWidth(4);
	for (int i=0; i<parametres->getPoses().count(); i++) {
		if (info && i==infoBulle.second)
			qglColor(QColor(255,0,0));
		else if (ref && i==infoBulle.second)
			qglColor(QColor(255,255,0));
		else
			qglColor(couleurs.at(i));
		QVector<Pt3dr> emprise = parametres->getPoses().at(i).getEmprise();
		glBegin(GL_LINES);
		for (int j=0; j<4; j++) {
                        glVertex3f(emprise.at(j).x, emprise.at(j).y, emprise.at(j).z);
			if (j==3)
                                glVertex3f(emprise.at(0).x, emprise.at(0).y, emprise.at(0).z);
			else
                                glVertex3f(emprise.at(j+1).x, emprise.at(j+1).y, emprise.at(j+1).z);
		}
		glEnd();
		glLineWidth(1);	
	}

	glEndList();
	return list;
}


GLuint GLWidget::makeObjectApp() {
//dessin des points de liaison 
	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);

	glPointSize(5);
	int n=0;
	if ((info || ref) && infoBulle.second!=-1) {
		if (info)
			qglColor(QColor(255,0,0,255));
		else
			qglColor(QColor(255,255,0,255));
		glBegin(GL_POINTS);
		const QList<pair<Pt3dr, QColor> >* l = &(parametres->getPoses().at(infoBulle.second).getPtsAppui());
		if (l->count()>0) {
			for (int j=0; j<l->count(); j++) {
				GLdouble xp = l->at(j).first.x;
				GLdouble yp = l->at(j).first.y;
				GLdouble zp = l->at(j).first.z;
				glVertex3d(xp, yp, zp);
				n++;			
			}
		}
//			QList<pair<Pt3dr, QColor> >* l2 = parametres->getPoses().at(infoBulle.second).getPtsAppui2nd();	//on allume aussi les points secondaires
//			for (int j=0; j<l2->count(); j++) {
//					GLdouble xp = l2->at(j).first.x;
//					GLdouble yp = l2->at(j).first.y;
//					GLdouble zp = l2->at(j).first.z;
//					glVertex3d(xp, yp, zp);
//			}
		glEnd();
	}
		
	glBegin(GL_POINTS);
	for (int i=0; i<parametres->getPoses().count(); i++) {
		if ((info || ref) && infoBulle.second!=-1 && infoBulle.second==i) continue;
		const QList<pair<Pt3dr, QColor> >* points = &(parametres->getPoses().at(i).getPtsAppui());
		if (points->count()>0) {
			for (int j=0; j<points->count(); j++) {
				GLdouble xp = points->at(j).first.x;
				GLdouble yp = points->at(j).first.y;
				GLdouble zp = points->at(j).first.z;
		                if (((info || ref) && infoBulle.second!=-1) || parametres->getColor()==0)	//blanc
					qglColor(QColor(255,255,255,255));
		                else if (parametres->getColor()==1) {	//hypso
					double teinte = (zp - parametres->getZoneChantier().at(4)) / (parametres->getZoneChantier().at(5) - parametres->getZoneChantier().at(4));
					teinte = 1-max (min (teinte,1.0), 0.0);
					QColor couleur;
					couleur.setHsv(teinte*300, 200, 255);
					qglColor(couleur);
		                } else if (parametres->getColor()==2) {	//texture
					qglColor(points->at(j).second);
				}
				glVertex3d(xp, yp, zp);
				n++;			
			}
		}
	}
	glEnd();
	glPointSize(1);

	glEndList();
	return list;
}

GLuint GLWidget::makeObjectNuag(int num, int etape) {
//dessin du num_ième nuage
        if (parametres->getNuaglayers().count()==0 || !parametres->getNuaglayers().contains(true)) return 0;
	if (etape>parametres->getNuages().at(num).getZoomMax()) return 0;

	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);
	
	glPointSize(1);

	glBegin(GL_POINTS);
	int n=0;
	const cElNuage3DMaille* nuage = parametres->getNuages().at(num).getPoints().at(etape);
	cElNuage3DMaille::tIndex2D it=nuage->Begin();

	QImage image(parametres->getNuages().at(num).getImageCouleur());	//image tif non tuilée
	if (!QFile(parametres->getNuages().at(num).getImageCouleur()).exists() || image.isNull() || image.size()==QSize(0,0))
		cout << tr("Fail to read image %1.\n").arg(parametres->getNuages().at(num).getImageCouleur()).toStdString();

	while(it!=nuage->End()) {
		//masque du nuage
		if (!nuage->IndexHasContenu(it)) {
			nuage->IncrIndex(it);
			continue;
		}
		Pt2dr pt = nuage->Index2Plani(Pt2dr(it.x,it.y));
		pt = parametres->getNuages().at(num).getGeorefMNT().terrain2Image(pt);
		//point 3D
		Pt3dr P = nuage->PtOfIndex(it);
		//coloration
		if (parametres->getColor()==GLParams::Hypso) {
			GLdouble teinte = (P.z - parametres->getZonenuage().at(4)) / (parametres->getZonenuage().at(5) - parametres->getZonenuage().at(4));
			teinte = 1-max (min (teinte,1.0), 0.0);
			QColor couleur;
			couleur.setHsv(teinte*359, 255, 255);	//couleur hypso
			qglColor(couleur);		
                } else if (parametres->getColor()==GLParams::Texture) {
			qglColor(QColor(image.pixel(pt.x,pt.y)));
                } else if (parametres->getColor()==GLParams::Mono) {	//monochrome (blanc)
			qglColor(QColor(255,255,255));	
		}
		glVertex3d(P.x, P.y, P.z);
		nuage->IncrIndex(it);
		n++;
	}
	glEnd();
	//delete correlImage;
	#if defined Q_WS_MAC
		if (n>2500000) parametres->modifNuages()[num].setZoomMax(etape);
	#endif

	glEndList();
	return list;
}

GLuint GLWidget::makeAxes() {
	//dessin des axes
	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);

	for (int i=0; i<3; i++) {
		QVector<int> axe(3,0);
		axe[i] = 1;
                qglColor(QColor(255*axe.at(0)+150*axe.at(2),255*axe.at(1)+150*axe.at(2),255*axe.at(2)));
		glPushMatrix();
		//glRotated(90+90*axe[2], -axe[1], axe[0]-axe[2], 0);
                if (i!=2) glRotated(90, -axe.at(1), axe.at(0), 0);	//initialement, les cylindres sont orientés vers +Z
		GLUquadric* cylindre = gluNewQuadric();
		gluCylinder(cylindre, 0.1, 0.1, 1, 10, 10);
		gluDeleteQuadric(cylindre);
		GLUquadric* cone = gluNewQuadric();
		glTranslatef(0, 0, 1.0);
		gluCylinder(cone, 0.2, 0, 1, 10, 10);
		gluDeleteQuadric(cone);
                glPopMatrix();
	}

	glEndList();
	return list;
}

GLuint GLWidget::makeBoule() {
	//dessin des cercles de la boule
	//GLuint list = glGenLists(1);
	parametres->incrNbGLLists();
	GLuint list = parametres->getNbGLLists();
	glNewList(list, GL_COMPILE);
	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();
                glCircle3i(1, parametres->modifRot());

		GLdouble * m = new GLdouble[16];	//pb avec glGet
		for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++) {
			if (j==1) m[i+4*j] = parametres->getRot()[i+4*(j+1)];
			else if (j==2) m[i+4*j] = -parametres->getRot()[i+4*(j-1)];
			else m[i+4*j] = parametres->getRot()[i+4*j];
		}}
		glRotated(90, 1, 0, 0);
		glCircle3i(1, m);

		GLdouble * n = new GLdouble[16];
		for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++) {
			if (j==0) n[i+4*j] = -m[i+4*(j+2)];
			else if (j==2) n[i+4*j] = m[i+4*(j-2)];
			else n[i+4*j] = m[i+4*j];
		}}
		glRotated(90, 0, 1, 0);
		glCircle3i(1, n);
	glPopMatrix();

	delete [] m;
	delete [] n;
	glEndList();
	return list;
}
void GLWidget::glCircle3i(GLint radius, GLdouble * m) {
	//dessine un cercle de rayon 1 et d'axe z
	//m : orientation finale du cercle pour dessiner l'arrière-plan en plus foncé
		//recherche des angles limites d'arrière-plan
	int precision = 100;
	double i1 = 0;
	double i2 = 0;
	double lim = (m[6]!=0)? -atan(-m[2]/m[6]) : PI/2.0;	//R31/R32
	if (m[6]>0) {
		i1 = lim*precision/2/PI;
		i2 = lim*precision/2/PI + 50;
	} else if (m[6]<0) {
		i1 = lim*precision/2/PI - 50;
		i2 = lim*precision/2/PI;
	} else if (m[6]==0 && -m[2]>0) {
		i1 = 25;
		i2 = 75;
	} else if (m[6]==0 && -m[2]<0) {
		i1 = -25;
		i2 = 25;
	} else if (m[6]==0 && m[2]==0) {
		i1 = 0;
		i2 = precision;
	}

		//dessin
	glDisable(GL_TEXTURE_2D);
	glLineWidth(3);
	glBegin(GL_LINE_LOOP);
		double angle;
		qglColor(QColor(155,200,255));
		for (int i=i1; i <i2; i++) {
			angle = i*2*PI/precision;
			glVertex2f(cos(angle)*radius, sin(angle)*radius);
		}
		if (m[6]!=0 || m[2]!=0) {	//(m[6]==0 && m[2]==0) => cercle d'axe z : tous les points sont visibles
			qglColor(QColor(78,100,128));
			if (i2>=100) i2 -= 100;
			else i1 += 100;	//i1<0
			for (int i=i2; i <i1; i++) {
				angle = i*2*PI/precision;
				glVertex2f(cos(angle)*radius, sin(angle)*radius);
			}
		}
	glEnd();
	glLineWidth(1);
}

void GLWidget::addNuages() {
        parametres->modifNuaglayers().resize(parametres->getNuages().count());
        parametres->modifNuaglayers().fill(true);
	currentDezoom.resize(parametres->getNuages().count());
	currentDezoom.fill(-1);
	for (int i=0; i<parametres->getNuages().count(); i++)
		calcDezoom(i);
	//à supprimer
	/*objectNuag.resize(6*parametres->getNuages().count());
	for (int i=0; i<parametres->getNuages().count(); i++) {
		for (int j=0; j<6; j++)
			objectNuag[6*i+j] = makeObjectNuag(i,j);
	}*/

	//redimensionnement de la projection pour coller aux nuages
     /*   GLdouble longueurChantier = max(max(parametres->getZonenuage().at(1)-parametres->getZonenuage().at(0),parametres->getZonenuage().at(3)-parametres->getZonenuage().at(2)),parametres->getZonenuage().at(5)-parametres->getZonenuage().at(4));
	espace[4] = 1;
	espace[5] = 1000*longueurChantier;
        GLdouble em = longueurChantier/(espace.at(5)+espace.at(4));
	espace[0] = -em;	//pour que le chantier prenne toute la fenêtre à une profondeur moyenne et une focale moyenne
	espace[1] = em;
	espace[2] = -em;
	espace[3] = em;*///la projection sera redimensionnée par vueChantier->exec => resiezGL => setupViewport	
}

//mise à jour///////////////////////////////////////////////////////
void GLWidget::paintEvent(QPaintEvent*) {
	resizeGL(width(), height());
	makeCurrent();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glLoadIdentity();

	//dessin des axes
        GLdouble sc = min(espace.at(1)-espace.at(0),espace.at(3)-espace.at(2))/10.0;
	glPushMatrix();
        glTranslated(espace.at(0)+sc, espace.at(2)+sc, -espace.at(4)-sc);
	glMultMatrixd(parametres->getRot());
	glScaled(sc/2,sc/2,sc/2);
	glCallList(axes);
	GLdouble * m = new GLdouble[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	GLdouble* m2 = new GLdouble[16];
	for (int j=0; j<4; j++) {
		for (int k=0; k<4; k++) {
			m2[4*j+k] = m[4*k+j];
		}
	}
	glPopMatrix();

	//dessin de la boule
        GLdouble sc2 = min(espace.at(1)-espace.at(0),espace.at(3)-espace.at(2))/espace.at(4)*(espace.at(5)+espace.at(4))/8.0;
	glPushMatrix();
        glTranslated(0, 0, -(espace.at(4)+espace.at(5))/2);
	glMultMatrixd(parametres->getRot());
	glScaled(sc2,sc2,sc2);
	glCallList(boule);
	glPopMatrix();

	//dessin du chantier centré (par rapport au viewport et à l'espace)
	//glTranslated(-(zoneChantierEtCam[0]+zoneChantierEtCam[1])/2.0, -(zoneChantierEtCam[2]+zoneChantierEtCam[3])/2.0, -(zoneChantierEtCam[4]+zoneChantierEtCam[5])/2.0-(espace[5]+espace[4])/2.0-abs(zoneChantierEtCam[5]-zoneChantierEtCam[4])/2.0);	//-(zoneChantierEtCam[5]-zoneChantierEtCam[4])/2.0 : le chantier est centré par rapport à son |z|min (sinon on voit pas tout)
        glTranslated(0, 0, -(espace.at(5)+espace.at(4))/2.0);
					//- + - : rotation de 180° autour de Y : X et Z négatifs => le chantier tourne autour de la caméra maîtresse
	glScaled(parametres->getScale(),parametres->getScale(),parametres->getScale());
	glMultMatrixd(parametres->getRot());
	glTranslated(parametres->getTrans().at(0), parametres->getTrans().at(1), parametres->getTrans().at(2));	// les transfo sont appliquées à l'envers : M'=MoR puis M'=MoT (dans le réf observateur)
        if (parametres->getCamlayers().at(0)) glCallList(objectCam);
        if (parametres->getCamlayers().at(2)) glCallList(objectEmp);
        if (parametres->getCamlayers().at(1)) glCallList(objectApp);
        if (parametres->getNuages().count()>0) {
                for (int i=0; i<parametres->getNuages().count(); i++) {
                        if (parametres->getNuaglayers().at(i))  {
				calcDezoom(i);
                                glCallList(objectNuag.at(6*i + currentDezoom.at(i)));
			}
		}
	}

	QPainter painter(this);	
	//affichage des info s'il y a lieu
	if ((info || ref) && infoBulle.second!=-1) {
		QString text = parametres->getPoses().at(infoBulle.second).getNomImg();
		QPoint P = infoBulle.first;
		QFontMetrics metrics = QFontMetrics(font());
		QRect rect = metrics.boundingRect(text);
		QPoint M(min(rect.width()+P.x(), width()), min(rect.height()+P.y()+10, height()));
		//painter.fillRect(QRect(M.x()-rect.width(), M.y()-rect.height(), rect.width(), rect.height()), QColor(255, 255, 255, 127));
		painter.fillRect(QRect(M.x()-rect.width(), M.y()-rect.height(), rect.width(), rect.height()), QColor(255, 255, 255, 255));
		painter.setPen(Qt::black);
		painter.drawText(M.x()-rect.width(), M.y()-rect.height(), rect.width(), rect.height(), Qt::AlignCenter | Qt::TextWordWrap, text);
	} 
	//label des axes
	GLint * view = new GLint[4];
	glGetIntegerv(GL_VIEWPORT, view);
	for (int i=0; i<3; i++) {
                QVector<int> axe(4,0);
				//axe[j] = (i==2)? -2.1 : 2.1;
                axe[i] = 2.1;
		axe[3] = 1.0;
                QVector<double> P(4,0);
                for (int j=0; j<4; j++) {
                        for (int k=0; k<4; k++)
                                P[j] += m2[4*j+k]*axe.at(k);
		}
                for (int j=0; j<3; j++)
                        P[j] /= P.at(3);
                for (int j=0; j<2; j++)
                        P[j] *= abs(espace.at(4)/P.at(2)*view[2]/(espace.at(1)-espace.at(0)));
		P[0] += view[0] + view[2]/2;
                P[1] = view[3] + view[1] - (P.at(1) + view[3]/2);
                QColor c(255*axe.at(0)/2.1+150*axe.at(2)/2.1,255*axe.at(1)/2.1+150*axe.at(2)/2.1,255*axe.at(2)/2.1);
		QString text = (i==0)? QString("X") : (i==1)? QString("Y") : QString("Z");
		QFontMetrics metrics = QFontMetrics(font());
		QRect rect = metrics.boundingRect(text);
		QPen pen(c);
		pen.setWidth(3);
		painter.setPen(pen);
                painter.drawText(P.at(0)-rect.width()/2, P.at(1)-rect.height()/2, rect.width(), rect.height(), Qt::AlignCenter, text);
	}
	painter.end(); 
	delete[] m; 
	delete[] m2; 
	delete[] view; 
}

bool GLWidget::calcDezoom(int num) {
	//recherche la bonne résolution pour le nuage, renvoie true si elle a changé
        if (parametres->getNuages().count()==0) return false;

	//points extrêmes du nuage
	QVector<QVector<REAL> > ptsExtremes(6, QVector<REAL>(3));	//+8 sommets ?
	for (int i=0; i<6; i++) {
		for (int j=0; j<3; j++)
			ptsExtremes[i][j] = (parametres->getNuages().at(num).getZone().at(2*j+1) + parametres->getNuages().at(num).getZone().at(2*j)) / 2;
	}
	for (int i=0; i<6; i+=2) {
		ptsExtremes[i][i/2] = parametres->getNuages().at(num).getZone().at(i);
		ptsExtremes[i+1][i/2] = parametres->getNuages().at(num).getZone().at(i+1);
	}

	//paramètres du nuage
        double f = double(parametres->getNuages().at(num).getFocale());	//en pixels
        const Pose* pose = &(parametres->getNuages().at(num).getPose());
    //-- const cElNuage3DMaille* nuage = parametres->getNuages().at(num).getPoints().at(0);
	//-- Pt2dr Pt = nuage->Plani2Index(Pt2dr(pose->width()/2,pose->height()/2));
	//double profondeurMoy = nuage->ProfEnPixel(Pt2di(Pt.x,Pt.y));	//peut aussi être la plus petite distance entre S et les ptsExtremes (au cas où l'objet ne serait pas centré)

	//distance des points au sommet pour avoir la profondeur réelle du nuage
	double profondeurMoy = numeric_limits<REAL>::max();
	Pt3dr C = pose->centre();
	QVector<REAL> C2;
	C2 << C.x << C.y << C.z;
	for (int k=0; k<6; k++) {
		double dist = 0;
		for (int i=0; i<3; i++)
			dist += (ptsExtremes[k][i]-C2[i]) * pose->rotation()(i,2);
			dist += (ptsExtremes[k][2]-C2[2]) * pose->rotation()(2,2);
		if (abs(profondeurMoy)>abs(dist)) profondeurMoy = dist;
	}

	//calcul de la profondeur des points extrêmes nuage à l'écran
	REAL Tz = -numeric_limits<REAL>::max();
	for (int k=0; k<6; k++) {
		REAL Tzecran = 0;
		QVector<REAL> P(ptsExtremes[k]);
		P.push_back(1);	
		GLdouble * m = new GLdouble[16];	//matrice terrain -> opengl (T0 s R T)
		glPushMatrix();
		glLoadIdentity();
		glTranslated(0, 0, -(espace.at(5)+espace.at(4))/2.0);
		glScaled(parametres->getScale(),parametres->getScale(),parametres->getScale());
		glMultMatrixd(parametres->getRot());
		glTranslated(parametres->getTrans().at(0), parametres->getTrans().at(1), parametres->getTrans().at(2));
		glGetDoublev(GL_MODELVIEW_MATRIX, m);
		glPopMatrix();

		double norm = 0;
		for (int i=0; i<4; i++) {
		        Tzecran += m[4*i+2] * P.at(i);
		        norm += m[4*i+3] * P.at(i);
		}
		Tzecran /= norm;
		delete [] m;
		Tzecran += -(espace.at(5)+espace.at(4))/2.0;
		if (abs(Tz)>abs(Tzecran)) Tz = Tzecran;
	}

	//dézoom (pow2 de 1 à 32)
	double dezoom = abs(Tz) * f * 2.0 / winZ / abs(profondeurMoy) / parametres->getScale();	// / (parametres->getFocale()/500.0) : compris dans winZ
	int newDezoom = floor(5.0 - log(max(0.0,dezoom))/log(2.0)) + 1;
	//int newDezoom = ceil(5.0 - log(max(0.0,dezoom))/log(2.0));	//pour avoir 1 pt/px de l'écran
	newDezoom = max (min (5, newDezoom), 0);
	newDezoom = min(newDezoom,parametres->getNuages().at(num).getZoomMax());

        if (newDezoom!=currentDezoom.at(num)) {
		currentDezoom[num] = newDezoom;
		return true;
	}
	return false;
}

void GLWidget::resizeGL(int width, int height) {
	setupViewport(width, height);
}

void GLWidget::setupViewport(int width, int height) {
	if (width==0 || height==0) return;
        int h = qMin(double(height), double(width)*(espace.at(3)-espace.at(2))/(espace.at(1)-espace.at(0)));
        int w = h*(espace.at(1)-espace.at(0))/(espace.at(3)-espace.at(2));
	glViewport((width - w) / 2, (height - h) / 2, w, h);
        winZ = w * espace.at(4) / (espace.at(1)-espace.at(0));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
        glFrustum(espace.at(0), espace.at(1), espace.at(2), espace.at(3), espace.at(4), espace.at(5));
     //   glFrustum(-espace.at(0)/2, espace.at(0)/2, -espace.at(2)/2, espace.at(2)/2, espace.at(4), espace.at(5));
	glMatrixMode(GL_MODELVIEW);
}

//changement de point de vue///////////////////////////////////////////////////:
void GLWidget::setRotation(GLdouble* R) {
        parametres->modifRot() = R;
	//boule = makeBoule();
	update();
}

void GLWidget::setTranslation(const QVector<GLdouble>& T) {
        if (T.at(0)==parametres->getTrans().at(0) && T.at(1)==parametres->getTrans().at(1) && T.at(2)==parametres->getTrans().at(2)) return;

	//si la translation est > à la moitié de la fenêtre + la moitié de l'objet, ça ne sert à rien
        QVector<GLdouble> Tverif(3);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(parametres->getRot());
	GLdouble * m = new GLdouble[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	for (int i=0; i<3; i++) {
	for (int j=0; j<3; j++) {
                Tverif[i] = m[j*4+i] * T.at(j);	//m'
	}}	
	glPopMatrix();
	/*double l = max(max(zoneChantierEtCam[1]-zoneChantierEtCam[0], zoneChantierEtCam[3]-zoneChantierEtCam[2]), zoneChantierEtCam[5]-zoneChantierEtCam[4])*sqrt(2)/2.0;
	for (int i=0; i<3; i++) {
		double tmax = l/2.0 + (espace[2*i+1] - espace[2*i])/2.0;
		if (abs(Tverif[i])>tmax) return;
	}*/

	for (int i=0; i<3; i++)
                parametres->modifTrans()[i] = T.at(i);
	update();
}

void GLWidget::convertRotation(int direction, const GLdouble& R, bool anti) {	//0 à 2
	//!anti : rotation selon la sphère, anti : rotation selon l'écran
	GLdouble* Rf = new GLdouble[16];
        QVector<GLdouble> axe(3,0);
	axe[direction] = 1;

	//la rotation dépend de la rotation courante
	glPushMatrix();
	glLoadIdentity();
	if (anti)
                glRotated(R, axe.at(0), axe.at(1), axe.at(2));
	glMultMatrixd(parametres->getRot());
	if (!anti)
                glRotated(R, axe.at(0), axe.at(1), axe.at(2));
	glGetDoublev(GL_MODELVIEW_MATRIX, Rf);
	glPopMatrix();

	setRotation(Rf);
}
void GLWidget::convertTranslation(int direction, const GLdouble& T) {	//0 à 2
        QVector<GLdouble> Tf(3);

	//la direction finale de la translation dépend de la rotation courante
	for (int i=0; i<3; i++) {
		Tf[i] = T * parametres->getRot()[i*4+direction] / parametres->getScale();
	}	
	glPopMatrix();

	for (int i=0; i<3; i++)
                Tf[i] = parametres->getTrans().at(i) + Tf.at(i);
        setTranslation(Tf);
}

void GLWidget::setScale(const GLdouble& sc) {	//0 à 2
        parametres->setScale(sc);
	update();
}
void GLWidget::multScale(int diff) {
        setScale(parametres->getScale()*(1.0+diff/20.0*parametres->getMaxScale()/parametres->getFocale()));	//+/- 20%, dépend de la focale
}

//récupération des signaux des outils////////////////////////////////////
void GLWidget::reinit() {
        QVector<GLdouble> T(3,0);
	setTranslation(T);

	glPushMatrix();
	glLoadIdentity();
	GLdouble * R = new GLdouble[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, R);
	glPopMatrix();
	setRotation(R);
	setScale(1);

	emit (changeFocale(parametres->getMaxScale()));
}

void GLWidget::translate(int direction) {	//1 à 6
	GLdouble diffT;
	int dir = (direction-1)/2;
	if (dir!=2) {
		double Tz = 0;
		for (int i=0; i<3; i++)
			Tz += parametres->getRot()[4*2+i] * parametres->getTrans().at(i);	//profondeur actuelle dans l'espace
                double dist = (espace.at(2*dir+1)-espace.at(2*dir)) / espace.at(4) * abs(Tz -(espace.at(5)+espace.at(4))/2.0);//largeur de l'espace à cette profondeur
		diffT = pow(double(-1),direction) * 0.05 * dist;// * 1.0/parametres->getScale() * parametres->getMaxScale()/focale
	} else
                diffT = pow(double(-1),direction) * 0.05 * (espace.at(5)-espace.at(1)) * parametres->getFocale()/parametres->getMaxScale();	//20 iterations pour parcourir l'espace
	convertTranslation(dir, diffT);// * 1.0/parametres->getScale() * parametres->getMaxScale()/focale
}

void GLWidget::rescale(int direction) {	//1 ou 2
	multScale(2*direction-3);
}

void GLWidget::modifFocale(int value) {	//0 à 1000
	double val = min( max(10.0,double(value)) ,parametres->getMaxScale()-1);
        if (val==parametres->getFocale()) return;
	//espace[4] = val/parametres->getMaxScale();
	GLdouble longueurChantier;
    //    if (parametres->getNuages().count()==0)
                longueurChantier = max(max(parametres->getZoneChantierEtCam().at(1)-parametres->getZoneChantierEtCam().at(0),parametres->getZoneChantierEtCam().at(3)-parametres->getZoneChantierEtCam().at(2)),parametres->getZoneChantierEtCam().at(5)-parametres->getZoneChantierEtCam().at(4));
	//else
      //          longueurChantier = max(max(parametres->getZonenuage().at(1)-parametres->getZonenuage().at(0),parametres->getZonenuage().at(3)-parametres->getZonenuage().at(2)),parametres->getZonenuage().at(5)-parametres->getZonenuage().at(4));
	
        GLdouble em = longueurChantier/(espace.at(5)+espace.at(4));
	for (int i=0; i<4; i++)
		espace[i] = parametres->getMaxScale()/val * -pow(double(-1),i)*em;
	setupViewport(width(), height());
	
	//modification de la profondeur pour que le chantier reste de même taille apparente
		//récupération de la profondeur
	/*GLdouble Tz = 0;
	for (int i=0; i<3; i++)
		Tz += parametres->getRot()[4*2+i] * parametres->getTrans()[i];
	Tz *= parametres->getScale();
		//mise à l'échelle
	Tz = (Tz-(espace[5]+espace[4])/2.0) * (val/focale - 1);
	convertTranslation(2, Tz);*/
        parametres->setFocale(val);
	update();
}

void GLWidget::rotate(int rotation, double angle) {
	if (rotation==0) return;
	convertRotation((rotation-1)/2, pow(double(-1),rotation)*angle,true);	//rotation suivant l'écran
}

void GLWidget::setInfo(bool b) {
	info = b;
	ref = false;
	if (!b) {
		objectCam = makeObjectCam();
		objectApp = makeObjectApp();
		update();
	}
}

void GLWidget::setRef(bool b) {
	ref = b;
	info = false;
	if (!b) {
		objectCam = makeObjectCam();
		objectApp = makeObjectApp();
		update();
	}
}

void GLWidget::setColor(const GLParams::Couleur& couleur) {
	QApplication::setOverrideCursor( Qt::WaitCursor );
        parametres->setColor(couleur);
	objectApp = makeObjectApp();
        if (parametres->getNuages().count()>0) {
                /*for (int i=0; i<parametres->getNuages().count(); i++)
                        objectNuag[6*i + currentDezoom.at(i)] = makeObjectNuag(i, currentDezoom.at(i));*/
		doNuages();
	}
	update();
	if (QApplication::overrideCursor()!=0) QApplication::restoreOverrideCursor();
}

void GLWidget::setMesure(const GLParams::Mesure& measure) { parametres->setMesure(measure); }

void GLWidget::dispCamLayers(int layer, bool display) {
        parametres->modifCamlayers()[layer] = display;
	update();
}

void GLWidget::dispNuagLayers(int layer, bool display) {
        parametres->modifNuaglayers()[layer] = display;
	update();
}

//récupération des signaux du clavier/////////////////////////////////
void GLWidget::keyPressEvent(QKeyEvent *event) {
	switch (event->key()) {
		case Qt::Key_Up :	
			convertRotation(0, 10, true);
			break;
		case Qt::Key_Down :	
			convertRotation(0, -10, true);
			break;
		case Qt::Key_Left :	
			convertRotation(1, 10, true);
			break;
		case Qt::Key_Right :	
			convertRotation(1, -10, true);
			break;
		case Qt::Key_8 :
		case Qt::Key_H :
                        if (parametres->getNuages().count()>0)
                                convertTranslation(1, (parametres->getZoneChantierEtCam().at(3)-parametres->getZoneChantierEtCam().at(2))/10.0);
			else
                                convertTranslation(1, (parametres->getZonenuage().at(3)-parametres->getZonenuage().at(2))/10.0);
			break;
		case Qt::Key_2 :
		case Qt::Key_B :
                        if (parametres->getNuages().count()>0)
                                convertTranslation(1, -(parametres->getZoneChantierEtCam().at(3)-parametres->getZoneChantierEtCam().at(2))/10.0);
			else
                                convertTranslation(1, -(parametres->getZonenuage().at(3)-parametres->getZonenuage().at(2))/10.0);
			break;	
		case Qt::Key_4 :
		case Qt::Key_G :
                        if (parametres->getNuages().count()>0)
                                convertTranslation(0, -(parametres->getZoneChantierEtCam().at(1)-parametres->getZoneChantierEtCam().at(0))/10.0);
			else
                                convertTranslation(0, -(parametres->getZonenuage().at(1)-parametres->getZonenuage().at(0))/10.0);
			break;	
		case Qt::Key_6 :
		case Qt::Key_D :
                        if (parametres->getNuages().count()>0)
                                convertTranslation(0, (parametres->getZoneChantierEtCam().at(1)-parametres->getZoneChantierEtCam().at(0))/10.0);
			else
                                convertTranslation(0, (parametres->getZonenuage().at(1)-parametres->getZonenuage().at(0))/10.0);
			break;	
		case Qt::Key_7 :
		case Qt::Key_9 :
		case Qt::Key_L :		
			convertTranslation(2, -10.0);
			break;
		case Qt::Key_1 :
		case Qt::Key_3 :
		case Qt::Key_P :		
			convertTranslation(2, 10.0);
			break;
		case Qt::Key_Plus :	
			multScale(1);
			break;
		case Qt::Key_Minus :	
			multScale(-1);
			break;
		case Qt::Key_Space :	
			convertRotation(2, 10, true);
			break;
		case Qt::Key_Alt :
		case Qt::Key_AltGr :	
			convertRotation(2, -10, true);
			break;
		case Qt::Key_0 :
		case Qt::Key_O :	
			reinit();
			break;
		case Qt::Key_R :	
                        modifFocale(parametres->getFocale()-5);
			break;
		case Qt::Key_A :	
                        modifFocale(parametres->getFocale()+5);
			break;
	}
}

//récupération des signaux de la souris/////////////////////////////////////////
pair<QVector<double>,QVector<double> > GLWidget::getMouseDirection (const QPoint& P, GLdouble * matrice) const {
	//récupère la direction et le point d'origine d'un clic souris P en coordonnées chantier (selon matrice)
		//vecteur clic en coordonnées observateur
	GLint * view = new GLint[4];
	glGetIntegerv(GL_VIEWPORT, view);
	double taille = min(view[2], view[3]);
	double window_x = P.x()-view[0] - double(view[2])/2.0;
	double window_y = (view[3] - P.y()+view[1]) - double(view[3])/2.0;
        double x = espace.at(1) * window_x / (taille/2.0);
        double y = espace.at(3) * window_y / (taille/2.0);	
        QVector<double> ray_pnt(4,0);
            ray_pnt[3] = 1;
        QVector<double> ray_vec(4,0);
            ray_vec[0] = x;
            ray_vec[1] = y;
            ray_vec[2] =  -espace.at(4);
	delete view;

        QVector<double> ray_pnt2(4,0);
        QVector<double> ray_vec2(4,0);
        for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++) {
                        ray_pnt2[i] += matrice[j*4+i] * ray_pnt.at(j);	//P2=n*P, n est selon les colonnes
                        ray_vec2[i] += matrice[j*4+i] * ray_vec.at(j);
		}
	}
	double norm_vec2 = 0;
	for (int i=0; i<3; i++)
                norm_vec2 += ray_vec2.at(i) * ray_vec2.at(i);
	norm_vec2 = sqrt(norm_vec2);
	for (int i=0; i<3; i++)
		ray_vec2[i] /= norm_vec2;
	
        return pair<QVector<double>,QVector<double> >(ray_pnt2,ray_vec2);
}

QVector<double> GLWidget::getSpherePoint(const QPoint& mouse) const {
	//point sur la sphère pointé par la souris (ref de la sphère)
		//direction du pointeur
	GLdouble * m = new GLdouble[16];
        GLdouble sc = min(espace.at(1)-espace.at(0),espace.at(3)-espace.at(2))/espace.at(4)*(espace.at(5)+espace.at(4))/8.0;
	glPushMatrix();
	glLoadIdentity();
	glScaled(1.0/sc,1.0/sc,1.0/sc);
	multTransposeMatrix(parametres->getRot());
        glTranslated(0, 0, +(espace.at(4)+espace.at(5))/2);
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	glPopMatrix();
        pair<QVector<double>,QVector<double> > pa = getMouseDirection (mouse, m);
        QVector<double> P = pa.first;
        QVector<double> V = pa.second;
		//point sur la sphère
        QVector<double> C(3,0);
        QVector<double> diff(3);
	for (int i=0; i<3; i++)
                diff[i] = P.at(i) - C.at(i);	//diff = P-C
	double scal1 = 0;	//scal1 = <V;P-C>
	double D1 = 0;	//D1 = |P-C|²
	double D2 = 0;	//D2 = |V|²
	for (int i=0; i<3; i++) {
                scal1 += V.at(i) * diff.at(i);
                D1 += diff.at(i) * diff.at(i);
                D2 += V.at(i) * V.at(i);
        }
	double delta = scal1*scal1 - (D1-1.0f)*D2;
	if (delta<0) {
		return getPlanPoint(pa);
	}
	delete [] m;
	delta = sqrt(delta);
	double lambda1 = (-scal1 - delta)/D2;
	double lambda2 = (-scal1 + delta)/D2;	//point de l'autre côté de la sphère
	double lambda = (abs(lambda1)<abs(lambda2)) ? lambda1 : lambda2;	//c'est le plus proche de P
        QVector<double> M(3);
	for (int i=0; i<3; i++)
                M[i] = P.at(i) + lambda*V.at(i);	//M = P + lambda*C
	return M;
}

QVector<double> GLWidget::getPlanPoint(const pair<QVector<double>,QVector<double> >& direction) const {
	//direction du pointeur
        QVector<double> P = direction.first;
        QVector<double> V = direction.second;
	
	//équation du plan
        QVector<double> O(4,0);
                O[3] = 1;
        QVector<double> N(4,0);
                N[2] = 1;

	GLdouble * m = new GLdouble[16];
	glPushMatrix();
	glLoadIdentity();
	multTransposeMatrix(parametres->getRot());
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	glPopMatrix();

        QVector<double> O2(4,0);
        QVector<double> N2(4,0);
        for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++) {
                        O2[i] += m[j*4+i] * O.at(j);	//O2=m*O, m est selon les colonnes
                        N2[i] += m[j*4+i] * N.at(j);
		}
	}
	delete [] m;
	double norm_N2 = 0;
	for (int i=0; i<3; i++)
                norm_N2 += N2.at(i) * N2.at(i);
	norm_N2 = sqrt(norm_N2);
	for (int i=0; i<3; i++)
		N2[i] /= norm_N2;
	
	//intersection
	double scal1 = 0;
	double scal2 = 0;
	for (int i=0; i<3; i++) {
                scal1 += N2.at(i) * (P.at(i) - O2.at(i));
                scal2 += N2.at(i) * (V.at(i) - O2.at(i));
        }
        double lambda = -scal1/scal2;
        QVector<double> M(3);
	for (int i=0; i<3; i++)
                M[i] = P.at(i) + lambda * V.at(i);
	return M;
}

void GLWidget::addCam() {
	emit cameraSelected(parametres->getPoses().at(infoBulle.second).getNomImg());
}

void GLWidget::mousePressEvent(QMouseEvent *event) {
	//cas 1 : acceptation d'une image de référence ou d'une image pour la corrélation (refBox)
	if (ref && (event->buttons() & Qt::RightButton) && infoBulle.second!=-1) {
		//menu ajouter à la liste
		QMenu menu(this);
		menu.addAction(addCamAct);
		menu.exec(event->globalPos());		
	}

	//cas 2 : cas général (translation, rotation)
	if (info || ref) return;
	lastPos = event->pos();
	if (info || ref || (event->buttons() & Qt::RightButton)) return;

	//cas 2b : rotation de la sphère
	//récupération de la rotation à effectuer
		//récupération du point sur la sphère
        QVector<double> C(3,0);
        QVector<double> M = getSpherePoint(event->pos());

		//cercle le plus proche et angle initial
	int cercle = 0;
        double dist = abs(M.at(0)-C.at(0));
	for (int i=1; i<3; i++) {
                if (abs(M.at(i)-C.at(i))<dist) {
                        dist = abs(M.at(i)-C.at(i));
			cercle = i;	//c'est le cercle autour de l'axe i
		}
	}
	int coordy = cercle - 1;
	if (coordy<0) coordy += 3;
	int coordx = coordy - 1;
	if (coordx<0) coordx += 3;
        double angle = atan2(M.at(coordy)-C.at(coordy), M.at(coordx)-C.at(coordx))*180/PI;	//on ne peut pas avoir x=y=0
        posSphere = pair<int,double>(cercle,angle);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event) {
	if (info || ref) return;
	if (event->x()<0 || event->y()<0 || event->x()>width() || event->y()>height())
		return;
	if (event->buttons() & Qt::LeftButton) {
                //point sur la sphère
                QVector<double> C(3,0);
                QVector<double> M = getSpherePoint(event->pos());
                if (M.count()==0) return;
		//nouvel angle
		int coordy = posSphere.first - 1;
		if (coordy<0) coordy += 3;
		int coordx = coordy - 1;
		if (coordx<0) coordx += 3;
                double angle = atan2(M.at(coordy)-C.at(coordy), M.at(coordx)-C.at(coordx))*180/PI;
		convertRotation(posSphere.first, angle - posSphere.second, false);	//rotation suivant la sphère
	} else if (event->buttons() & Qt::RightButton) {
		double dx = event->x() - lastPos.x();
		double dy = event->y() - lastPos.y();
                GLdouble coeff = (-parametres->getTrans().at(2) + (espace.at(5)+espace.at(4))/2.0) / winZ;
		convertTranslation(0, dx*coeff);
		convertTranslation(1, -dy*coeff);
	}
	lastPos = event->pos();
}

void GLWidget::wheelEvent(QWheelEvent *event) {
	if (info || ref) return;
	if (event->delta()==0) return;
	double dz = event->delta() / abs(event->delta());
	multScale(dz);
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event) {
	if (!info && !ref) return;

	//recherche de la caméra correspondante
		//direction du pointeur
 /*	GLdouble * m = new GLdouble[16];
	glPushMatrix();
	glLoadIdentity();
//	glTranslated(-parametres->getTrans().at(0), -parametres->getTrans().at(1), -parametres->getTrans().at(2));
//	multTransposeMatrix(parametres->getRot());
//	glScaled(1.0/parametres->getScale(),1.0/parametres->getScale(),1.0/parametres->getScale());
        glTranslated(0, 0, +(espace.at(5)+espace.at(4))/2.0);
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	glPopMatrix();
        pair<QVector<double>,QVector<double> > p = getMouseDirection (event->pos(), m);
	delete [] m;
        QVector<double> ray_pnt2 = p.first;
        QVector<double> ray_vec2 = p.second;

		//recherche de la caméra
	int N = -1;
	GLdouble Zmax = -numeric_limits<double>::max();	//les Z sont négatifs
	GLdouble Dmin = numeric_limits<double>::max();
	for (int i=0; i<parametres->getPoses().count(); i++) {
		//if (profondeur(parametres->getPoses().at(i).centre())<Zmax)
		//	continue;
		//calcul de la distance au rayon
                QVector<double> M(3);
                        M[0] = parametres->getPoses().at(i).centre().x - ray_pnt2.at(0);
                        M[1] = parametres->getPoses().at(i).centre().y - ray_pnt2.at(1);
                        M[2] = parametres->getPoses().at(i).centre().z - ray_pnt2.at(2);
		double distance = 0;
		for (int j=0; j<3; j++)
                        distance += M.at(j) * ray_vec2.at(j);
		distance *= distance;
		for (int j=0; j<3; j++)
                        distance -= M.at(j) * M.at(j);
		distance = sqrt(-distance);
                REAL f = parametres->getPoses().at(i).getCamera().Focale();
		int w = parametres->getPoses().at(i).width();
		int h = parametres->getPoses().at(i).height();
                if (distance<parametres->getDistance()*parametres->getScale()*min(min(f,double(w)),double(h)) && profondeur(parametres->getPoses().at(i).centre())<Zmax) {
			N = i;
			Zmax = profondeur(parametres->getPoses().at(i).centre());
                        Dmin = parametres->getDistance()*parametres->getScale()*min(min(f,double(w)),double(h));
		}
                else if (distance<parametres->getDistance()*parametres->getScale()*max(f,sqrt(w*w+h*h)) && distance<Dmin) {
			N = i;
			Dmin = distance;
		}
	}*/

	//2e méthode : projection des caméras sur l'écran, calcul de la distance à l'utilisateur et sélection de la caméra la + proche du clic (et si incertitude, de l'utilisateur)GLdouble * m = new GLdouble[16];
 	GLdouble * m = new GLdouble[16];
	glPushMatrix();
		glLoadIdentity();
		glTranslated(0, 0, -(espace.at(5)+espace.at(4))/2.0);
		glScaled(parametres->getScale(), parametres->getScale(), parametres->getScale());
		glMultMatrixd(parametres->getRot());
		glTranslated(parametres->getTrans().at(0), parametres->getTrans().at(1), parametres->getTrans().at(2));
		glGetDoublev(GL_MODELVIEW_MATRIX, m);
	glPopMatrix();

	GLint * view = new GLint[4];
	glGetIntegerv(GL_VIEWPORT, view);
	double taille = min(view[2], view[3]);	//taille réelle de la fenêtre de visibilité sur l"espace 3D

	QVector<QVector<GLdouble> > coordPlanes(parametres->getPoses().count(), QVector<GLdouble>(2,0) );	//x,y écran
	QVector<double> distances(parametres->getPoses().count());	//distance à l'utilisateur
	for (int i=0; i<parametres->getPoses().count(); i++) {
                QVector<double> M(4);
                        M[0] = parametres->getPoses().at(i).centre().x;
                        M[1] = parametres->getPoses().at(i).centre().y;
                        M[2] = parametres->getPoses().at(i).centre().z;
                        M[3] = 1.0;

		//coordonnées dans l'espace utilisateur
                QVector<double> M2(4,0.0);
		for (int j=0; j<4; j++) {
			for (int k=0; k<4; k++)
		                M2[j] += m[k*4+j] * M.at(k);	//M2=m*M, avec m selon les colonnes
		}

		double X = M2[0] / M2[3];
		double Y = M2[1] / M2[3];
		double Z = M2[2] / M2[3];

		//distance à l'utilisateur
		distances[i] = (Z<=-espace.at(4))? -Z : numeric_limits<double>::max();
		if (-Z<espace.at(4)) continue;	//caméra non visible

		//coordonnées écran
		double X2 = X * espace.at(4) / -Z;
		double Y2 = Y * espace.at(4) / -Z;

		double window_x = X2 * taille/2.0 / espace.at(1);
		double window_y = Y2 * taille/2.0 / espace.at(3);

		coordPlanes[i][0] = window_x + double(view[2])/2.0 + view[0];
		coordPlanes[i][1] = (double(view[3])/2.0 - window_y) + view[1];
	}
	delete [] m;
	delete [] view;

	QPoint clic = event->pos();
	int N = -1;
	GLdouble Zmax = numeric_limits<double>::max();
	GLdouble Dmin = numeric_limits<double>::max();
	for (int i=0; i<parametres->getPoses().count(); i++) {
		if (distances.at(i)==numeric_limits<double>::max()) continue;

		//distance au clic
		double distance = realDistance2(QPointF( clic.x()-coordPlanes.at(i).at(0) , clic.y()-coordPlanes.at(i).at(1) ));
		if (distance>30*30) continue;
		else if (distance>9 && distance<Dmin) {
			Dmin = distance;
			Zmax = numeric_limits<double>::max();
			N = i;
		} else if (distance<=9 && Dmin>9) {
			Dmin = distance;
			N = i;
			Zmax = distances.at(i);
		} else if (distance<=9 && Zmax>distances.at(i)) {
			N = i;
			Zmax = distances.at(i);
		}
		 if (distance<Dmin) {
			N = i;
			Dmin = distance;
		}
	}

	//affichage du nom
	infoBulle = QPair<QPoint,int>(event->pos(), N);	//si N==-1, suppression de la couleur
	objectCam = makeObjectCam();
	objectApp = makeObjectApp();
	update();
}
GLdouble GLWidget::profondeur(const Pt3dr& point) const { return parametres->getRot()[2] * point.x + parametres->getRot()[6] * point.y + parametres->getRot()[10] * point.z; }

//param/////////////////////////////////////////////////////////////
int GLWidget::getRefImg() const { return infoBulle.second; }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


VueChantier::VueChantier(const ParamMain* pMain, QWidget* parent, Assistant* help):
	QDialog(parent),
	glWidget(0),
	assistant(help),
	hidden(false),
	paramMain(pMain),
	glparams()
{
	done = false;
	setWindowModality(Qt::ApplicationModal);
	timer = new QTimer;
	timer->setSingleShot(true);
	resize(sizeHint());
	setMinimumSize(minimumSizeHint());
	setMaximumSize(maximumSizeHint());

	moveButtons = new QToolButton*[8];
	for (int i=0; i<8; i++)
		moveButtons[i] = new QToolButton;
	//boutons de translation
	translationBox = new QGroupBox(conv(tr("Translate scene")));
	intiViewButton = new QToolButton;
	intiViewButton->setIcon(QIcon(g_iconDirectory+"radialGradient.png"));
	intiViewButton->setToolTip(conv(tr("Initialize view")));
	moveButtons[0]->setIcon(QIcon(g_iconDirectory+"linguist-prev.png"));
	moveButtons[0]->setToolTip(conv(tr("Move left")));
	moveButtons[1]->setIcon(QIcon(g_iconDirectory+"linguist-next.png"));
	moveButtons[1]->setToolTip(conv(tr("Move right")));
	moveButtons[2]->setIcon(QIcon(g_iconDirectory+"linguist-down.png"));
	moveButtons[2]->setToolTip(conv(tr("Move dowm")));
	moveButtons[3]->setIcon(QIcon(g_iconDirectory+"linguist-up.png"));
	moveButtons[3]->setToolTip(conv(tr("Move up")));
	moveButtons[5]->setIcon(QIcon(g_iconDirectory+"linguist-diag2.png"));
	moveButtons[5]->setToolTip(conv(tr("Move closer")));	//Z négatif
	moveButtons[4]->setIcon(QIcon(g_iconDirectory+"linguist-diag.png"));
	moveButtons[4]->setToolTip(conv(tr("Move farer")));
	QGridLayout* translationLayout = new QGridLayout;
	translationLayout->addWidget(moveButtons[3], 0, 1, 1, 1,Qt::AlignHCenter);
	translationLayout->addWidget(moveButtons[4], 0, 2, 1, 1,Qt::AlignLeft);
	translationLayout->addWidget(moveButtons[0], 1, 0, 1, 1,Qt::AlignRight);
	translationLayout->addWidget(intiViewButton, 1, 1, 1, 1,Qt::AlignHCenter);
	translationLayout->addWidget(moveButtons[1], 1, 2, 1, 1,Qt::AlignLeft);
	translationLayout->addWidget(moveButtons[5], 2, 0, 1, 1,Qt::AlignRight);
	translationLayout->addWidget(moveButtons[2], 2, 1, 1, 1,Qt::AlignHCenter);
	translationBox->setLayout(translationLayout);

	zoomBox = new QGroupBox(tr("Zoom in"));
	moveButtons[6]->setIcon(QIcon(g_iconDirectory+"designer-property-editor-remove-dynamic2.png"));
	moveButtons[6]->setToolTip(conv(tr("Scale down")));
	moveButtons[7]->setIcon(QIcon(g_iconDirectory+"designer-property-editor-add-dynamic.png"));
	moveButtons[7]->setToolTip(tr("Scale up"));
	QHBoxLayout* zoomLayout = new QHBoxLayout;
	zoomLayout->addWidget(moveButtons[7],0,Qt::AlignRight);
	zoomLayout->addWidget(moveButtons[6],0,Qt::AlignLeft);
	zoomBox->setLayout(zoomLayout);

	focaleBox = new QGroupBox;
	QToolButton* focaleButton = new QToolButton ;
	focaleButton->setIcon(QIcon(g_iconDirectory+"movie_grey_camera.png"));
	focaleButton->setEnabled(false);
	focaleButton->resize(QSize(20,20));
	focaleButton->setToolTip(tr("View focal length"));
	focaleSlide = new QSlider;
	focaleSlide->setOrientation(Qt::Horizontal);
	focaleSlide->setMaximum(2*int(glparams.getMaxScale()));
	focaleSlide->setMinimum(0);
	focaleSlide->setSingleStep(int(glparams.getMaxScale())/100);
	focaleSlide->setTickPosition(QSlider::TicksBelow);
	focaleSlide->setValue(int(glparams.getMaxScale()));
	focaleSlide->setToolTip(tr("View focal length"));
	QHBoxLayout* focaleLayout = new QHBoxLayout;
	focaleLayout->addWidget(focaleButton,0,Qt::AlignLeft);
	focaleLayout->addWidget(focaleSlide);
	focaleBox->setLayout(focaleLayout);

	rotationBox = new QGroupBox(tr("Rotation"));
	QGridLayout* rotationLayout = new QGridLayout;
	rotationButton = new RotationButton;
	rotationLayout->addWidget(rotationButton,0,0,3,3,Qt::AlignHCenter);
	rotationBox->setLayout(rotationLayout);

	aperoLayers = new Layers;
	QVector<QString> noms;
	if (paramMain->isFrench()) noms.append(QApplication::translate("Dialog", "Cameras", 0, QApplication::CodecForTr));	//pb affichage avec conv(tr())
	else noms.append(QString("Cameras"));	//pb affichage avec conv(tr())
	noms.append(tr("3D tie-points"));
	noms.append(tr("Image bounding box"));
	aperoLayers->create(tr("Display"), noms);
	aperoLayers->setChecked(2,false);
	QGroupBox* affButtonsBox = new QGroupBox;
	QHBoxLayout* affButtonsLayout = new QHBoxLayout;
	infoButton = new QToolButton ;
	infoButton->setIcon(QIcon(g_iconDirectory+"qmessagebox-info.png"));
	infoButton->setToolTip(tr("Image name"));
	infoButton->setCheckable(true);
	infoButton->setChecked(false);
	colorButton = new QToolButton ;
	colorButton->setIcon(QIcon(g_iconDirectory+"color.png"));
	colorButton->setToolTip(conv(tr("texture\nDisplay points with hypsometric colours")));
	colorButton->setCheckable(true);
	colorButton->setChecked(true);
	measureButton = new QToolButton ;
	measureButton->setIcon(QIcon(g_iconDirectory+"measure.png"));
	measureButton->setToolTip(tr("Measure 3D distances"));
	measureButton->setCheckable(true);
	measureButton->setChecked(false);
	affButtonsLayout->addWidget(infoButton);
	affButtonsLayout->addWidget(colorButton);
	//affButtonsLayout->addWidget(measureButton);
	affButtonsLayout->addStretch();
	affButtonsBox->setLayout(affButtonsLayout);
	aperoLayers->addWidget(affButtonsBox,Qt::AlignHCenter);

	nuagesLayers = new Layers;

	refBox = new SelectCamBox;

	helpButton = new QToolButton ;
	helpButton->setIcon(QIcon(g_iconDirectory+"linguist-check-off.png"));
	helpButton->setToolTip(tr("Help"));

	mapper3 = new QSignalMapper(); 	
	for (int i=0; i<8; i++) {
		connect(moveButtons[i], SIGNAL(pressed()), mapper3, SLOT(map()));
		mapper3->setMapping(moveButtons[i], i);
	}
	connect(mapper3, SIGNAL(mapped(int)),this, SLOT(answerButton(int)));
	connect(timer, SIGNAL(timeout()),this, SLOT(answerButton()));
	connect(infoButton, SIGNAL(clicked()), this, SLOT(infoClicked()));
	connect(colorButton, SIGNAL(clicked()), this, SLOT(hypsoClicked()));
	connect(measureButton, SIGNAL(clicked()), this, SLOT(measureClicked()));
	connect(refBox, SIGNAL(refClicked()), this, SLOT(refClicked()));
	connect(refBox, SIGNAL(okClicked()), this, SLOT(okClicked()));
	connect(helpButton, SIGNAL(clicked()), this, SLOT(helpClicked()));

	toolBar1 = new QToolBar;
	toolBar1->setOrientation(Qt::Vertical);
	toolBar1->addWidget(translationBox);
	toolBar1->addWidget(zoomBox);
	toolBar1->addWidget(focaleBox);
	toolBar1->addWidget(rotationBox);
	toolBar1->addSeparator();

	toolBar2 = new QToolBar;
	toolBar2->setOrientation(Qt::Vertical);
	toolBar2->addWidget(aperoLayers);
	toolBar2->addWidget(nuagesLayers);
	toolBar2->addSeparator();
	toolBar2->addWidget(refBox);
	toolBar2->addSeparator();
	toolBar2->addWidget(helpButton);

	if (!getParamPoses()) return;
	glWidget = new GLWidget(this, &glparams, *paramMain);

	mainLayout = new QGridLayout;
	setLayout(mainLayout);
	setWindowTitle(tr("Survey view"));
	initialise();
	done = true;
}
VueChantier::~VueChantier() {
	if (glWidget!=0) delete glWidget;
	delete [] moveButtons;
	delete rotationButton;
	delete aperoLayers;
	delete nuagesLayers;
	delete mapper3;
	delete timer;
}

QSize VueChantier::sizeHint () { return QApplication::desktop()->availableGeometry().size(); }
QSize VueChantier::minimumSizeHint () { return QApplication::desktop()->availableGeometry().size()*2/3; }
QSize VueChantier::maximumSizeHint () { return QApplication::desktop()->availableGeometry().size(); }

bool VueChantier::getParamPoses () {
//récupère les paramètres de chaque caméra dans les fichiers Ori-F, les points homologues 3D et les paramètres du chantier
	glparams.setDossier(paramMain->getDossier());

	//lecture des caméras
	glparams.modifPoses().resize(paramMain->getParamApero().getImgToOri().count());
	QString err = convert (paramMain, glparams.modifPoses());
	if (!err.isEmpty()) {
		qMessageBox(this, tr("File error."),err);
		return false;
	}

	if (paramMain->getParamApero().getCalcPts3D()) {
		//lecture des points 3D
		for (int i=0; i<glparams.getPoses().count(); i++) {
			QFile file(paramMain->getDossier() + QString("Ori-F/") + QString("3D/") + glparams.getPoses().at(i).getNomImg().section(".",0,-2) + QString(".txt"));
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
				qMessageBox(this, tr("Read error."), conv(tr("Fail to open 3D point file %1.")).arg(file.fileName()));
				return false;
			}
			QTextStream inStream(&file);
			while (!inStream.atEnd()) {				
				QString line = inStream.readLine();
				QVector<double> P(3);
				for (int j=0; j<3; j++) {
					bool ok = false;
					P[j] = line.section(" ",j,j).toFloat(&ok);
					if (!ok) {
						qMessageBox(this, tr("Read error."), conv(tr("Fail to read a point in file %1\n%2")).arg(file.fileName()).arg(line));
						return false;
					}				
				}
		                Pt3dr pt3D(P.at(0),P.at(1),P.at(2));
				QVector<double> C(3);
				for (int j=3; j<6; j++) {
					bool ok = false;
					C[j-3] = line.section(" ",j,j).toFloat(&ok);
					if (!ok) {
						qMessageBox(this, tr("Read error."), conv(tr("Fail to read a point colour in file %1\n%2")).arg(file.fileName()).arg(line));
						return false;
					}				
				}
		                QColor color(C.at(0),C.at(1),C.at(2));
				glparams.modifPoses()[i].modifPtsAppui().push_back(pair<Pt3dr, QColor>(pt3D,color));
			}
			file.close();
		}
	}

	//récupération des autres paramètres
	QFile file2(paramMain->getDossier() + QString("Ori-F/") + QString("3D/") + QString("param_chantier")); 
	if (!file2.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qMessageBox(this, tr("Read error."), conv(tr("Fail to open parameter file %1.")).arg(file2.fileName()));
		return false;
	}
	QTextStream inStream2(&file2);
	bool ok = false;
	
	QString line = inStream2.readLine();
	while (line.isEmpty()) line = inStream2.readLine();
	double echelle = line.toFloat(&ok);
	if (!ok) {
		qMessageBox(this, tr("Read error."), conv(tr("Fail to read scale parameter.\n%1 : %2")).arg(file2.fileName()).arg(line));
		return false;		
	}	
	glparams.setDistance(echelle);
		
	line = inStream2.readLine();
	while (line.isEmpty()) line = inStream2.readLine();
	QVector<GLdouble> zoneChantier(6);
	for (int i=0; i<6; i++) {
		zoneChantier[i] = line.section(" ",i,i).toFloat(&ok);
		if (!ok) {
			qMessageBox(this, tr("Read error."), conv(tr("Fail to read scene bounding box parameters.\n%1 : %2")).arg(file2.fileName()).arg(line));
			return false;
		}				
	}
	glparams.setZoneChantier(zoneChantier);

	line = inStream2.readLine();
	while (line.isEmpty()) line = inStream2.readLine();
	QVector<GLdouble> zoneChantierEtCam(6);
	for (int i=0; i<6; i++) {
		zoneChantierEtCam[i] = line.section(" ",i,i).toFloat(&ok);
		if (!ok) {
			qMessageBox(this, tr("Read error."), conv(tr("Fail to read scene parameters including camera bounding box.\n%1 : %2")).arg(file2.fileName()).arg(line));
			return false;
		}				
	}
	glparams.setZoneChantierEtCam(zoneChantierEtCam);	
		
	for (int i=0; i<glparams.modifPoses().count(); i++) {
		QVector<Pt3dr> emprise(4);
		for (int j=0; j<4; j++) {
			line = inStream2.readLine();
			while (line.isEmpty()) line = inStream2.readLine();
			QVector<double> P(3);
			for (int k=0; k<3; k++) {
				P[k] = line.section(" ",k,k).toFloat(&ok);
				if (!ok) {
					qMessageBox(this, tr("Read error."), conv(tr("Fail to read image 3D bounding box parameters.\n%1 : %2")).arg(file2.fileName()).arg(line));
					return false;
				}
			}	
                        emprise[j] = Pt3dr(P.at(0),P.at(1),P.at(2));
		}
		glparams.modifPoses()[i].setEmprise(emprise);
	}
	file2.close();
	return true;
}

QString VueChantier::getHomol3D (const QString& nomPose, const QString& dossier, QList<pair<Pt3dr, QColor> >& listPt3D) {
//utilisé par drawMask (attention, les fichiers sont vides si l'option "calculer les points 3D" a été désactivée dans Apero)
	QFile file(dossier + QString("Ori-F/") + QString("3D/") + nomPose.section(".",0,-2) + QString(".txt"));
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		return conv(tr("Fail to open 3D point file %1.")).arg(file.fileName());
	}
	QTextStream inStream(&file);
	while (!inStream.atEnd()) {				
		QString line = inStream.readLine();
                QVector<double> P(3);
		for (int i=0; i<3; i++) {
			bool ok = false;
			P[i] = line.section(" ",i,i).toFloat(&ok);
			if (!ok) {
				return conv(tr("Fail to read a point in file %1\n%2")).arg(file.fileName()).arg(line);
			}				
		}
                Pt3dr pt3D(P.at(0),P.at(1),P.at(2));
                QVector<double> C(3);
		for (int i=3; i<6; i++) {
			bool ok = false;
			C[i-3] = line.section(" ",i,i).toFloat(&ok);
			if (!ok) {
				return conv(tr("Fail to read a point colour in file %1\n%2")).arg(file.fileName()).arg(line);
			}				
		}
                QColor color(C.at(0),C.at(1),C.at(2));
		listPt3D.push_back(pair<Pt3dr, QColor>(pt3D,color));
	}
	file.close();
	if (listPt3D.count()==0) return conv(tr("No 3D points found."));
	return QString();
}

void VueChantier::initialise () {
	mainLayout->addWidget(glWidget,0,0,2,1);
	mainLayout->addWidget(toolBar1,0,1,1,1);
	mainLayout->addWidget(toolBar2,1,1,1,1);
	connect(intiViewButton, SIGNAL(clicked()), glWidget, SLOT(reinit()));
	connect(focaleSlide, SIGNAL(valueChanged(int)), glWidget, SLOT(modifFocale(int)));
	connect(glWidget, SIGNAL(changeFocale(int)), this, SLOT(setFocale(int)));
	connect(rotationButton, SIGNAL(roll(int,double)), glWidget, SLOT(rotate(int,double)));
	connect(aperoLayers, SIGNAL(stateChanged(int,bool)), glWidget, SLOT(dispCamLayers(int,bool)));
	connect(this, SIGNAL(dispNuages()), glWidget, SLOT(addNuages()));
	connect(glWidget, SIGNAL(cameraSelected(QString)), refBox, SLOT(addCamera(QString)));
}

void VueChantier::hideEvent(QHideEvent*) { hidden = true; }
void VueChantier::show(const SelectCamBox::Mode& refMode, const QString& refImg, const QStringList& precCam) {
	QDialog::show();
	refBox->create(refMode, refImg, precCam);
}
void VueChantier::showEvent(QShowEvent*) {	//qglwidget se bloque à chaque réduction de fenêtre et les paramètres opengl ne sont plus définis
	if (!hidden) return;
	mainLayout->removeWidget(toolBar1);
	mainLayout->removeWidget(toolBar2);
	mainLayout->removeWidget(glWidget);
	delete glWidget;
	glWidget = new GLWidget(this, &glparams, *paramMain);
	initialise();	
	infoButton->setChecked(false);
}

void VueChantier::resizeEvent(QResizeEvent*) {
//pour les ordinateur à petit écran, pb d'affichage du bas de la barre d'outil -> on déplace la moitié à droite
	if (height()<translationBox->height()+zoomBox->height()+focaleBox->height()+rotationBox->height()+aperoLayers->height()+nuagesLayers->height()+refBox->height()+helpButton->height()+3*10) {
		mainLayout->removeWidget(toolBar1);
		mainLayout->removeWidget(toolBar2);
		mainLayout->removeWidget(glWidget);
		mainLayout->addWidget(glWidget,0,0,1,1);
		mainLayout->addWidget(toolBar1,0,1,1,1);
		mainLayout->addWidget(toolBar2,0,2,1,1);
	} else {
		mainLayout->removeWidget(toolBar1);
		mainLayout->removeWidget(toolBar2);
		mainLayout->removeWidget(glWidget);
		mainLayout->addWidget(glWidget,0,0,2,1);
		mainLayout->addWidget(toolBar1,0,1,1,1);
		mainLayout->addWidget(toolBar2,1,1,1,1);
	}	
}

void VueChantier::answerButton(int button) {
	if (button!=-1)
		currentTool = button;
	else {
		if (currentTool<6)
			glWidget->translate(currentTool+1);
		else
			glWidget->rescale(currentTool-5);
	}
	if (moveButtons[currentTool]->isDown()) {
		timer->start(250);
	}
}

void VueChantier::setFocale(int f) {
	focaleSlide->setValue(f);
}

void VueChantier::infoClicked() {
	refBox->setRefButtonChecked(false);
	measureButton->setChecked(false);
	glWidget->setInfo(infoButton->isChecked());
}

void VueChantier::textureClicked() {
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(textureClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(hypsoClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(noColorClicked()));
	colorButton->setIcon(QIcon(g_iconDirectory+"color.png"));
	colorButton->setToolTip(conv(tr("texture\nDisplay points with hypsometric colours")));
	connect(colorButton, SIGNAL(clicked()), this, SLOT(hypsoClicked()));
	glWidget->setColor(GLParams::Texture);
}

void VueChantier::hypsoClicked() {
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(textureClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(hypsoClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(noColorClicked()));
	colorButton->setIcon(QIcon(g_iconDirectory+"white.png"));
	colorButton->setToolTip(conv(tr("hypsometric colours\nDisplay points in grayscale")));
	connect(colorButton, SIGNAL(clicked()), this, SLOT(noColorClicked()));
	glWidget->setColor(GLParams::Hypso);
}

void VueChantier::noColorClicked() {
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(textureClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(hypsoClicked()));
	disconnect(colorButton, SIGNAL(clicked()), this, SLOT(noColorClicked()));
	colorButton->setIcon(QIcon(g_iconDirectory+"texture.png"));
	colorButton->setToolTip(conv(tr("grayscale\nDisplay points with texture")));
	connect(colorButton, SIGNAL(clicked()), this, SLOT(textureClicked()));
	glWidget->setColor(GLParams::Mono);
}

void VueChantier::measureClicked() {
	infoButton->setChecked(false);	
	refBox->setRefButtonChecked(false);
        GLParams::Mesure mesure = glparams.getMesure();
	switch (mesure) {
		case GLParams::Aucune :
			mesure = GLParams::S1;
			break;
		case GLParams::V1 :
			mesure = GLParams::S2;
			break;
		default :
			mesure = GLParams::Aucune;	//abandon
			break;
	}
	glWidget->setMesure(mesure);
}

void VueChantier::refClicked() {
	infoButton->setChecked(false);
	measureButton->setChecked(false);
	glWidget->setRef(refBox->isRefButtonChecked());
}

void VueChantier::okClicked() { 
	hide();
	accept();
}

void VueChantier::helpClicked() {
	if (glparams.getNuages().count()==0) assistant->showDocumentation(assistant->pageVueChantier);
	else  assistant->showDocumentation(assistant->pageVueNuages);
}

QString VueChantier::convert (const ParamMain* pMain, QVector<Pose>& cameras, int N) {
//lecture des paramètres des caméra dans les fichiers Ori-F
//si N!=-1, ne lit que la Nième caméra (utilisé pour le masque automatique)
	int i0 = (N==-1)? 0 : N;
	int i1 = (N==-1)? pMain->getParamApero().getImgToOri().count() : N+1;

	//système métrique
	QString virgule;
	QString err = systemeNumerique(virgule);
	if (!err.isEmpty())
		return err;

	//récupération des tailles des images (1 fois)
	/*err = pMain->saveImgsSize();
	if (!err.isEmpty())
		return err;*/
	for (int i=0; i<i1-i0; i++) {
		int idx = pMain->findImg(pMain->getParamApero().getImgToOri().at(i), 1);
		cameras[i].setImgSize(pMain->getCorrespImgCalib().at(idx).getTaille());
	}

	for (int i=i0; i<i1; i++) {
		//param initiaux + nom du fichier
		cameras[i].setNomImg(pMain->getParamApero().getImgToOri().at(i));
	
		//nom du fichier
		QString file = pMain->getDossier() + QString("Ori-F/") + QString("OrFinale-") + pMain->getParamApero().getImgToOri().at(i).section(".",0,-2) + QString(".xml");
		//conversion du fichier (pb de ,)
			//récupération des fichiers appelés (calibrations...)
		QFile oldFile(file);
		if (!oldFile.open(QFile::ReadOnly | QFile::Text))
				return conv(tr("Fail to read file %1.")).arg(file);
		QXmlStreamReader xmlReader(&oldFile);
		QString calibXml, calibXml2;
		while (!xmlReader.atEnd() && (!xmlReader.isStartElement() || xmlReader.name().toString()!=QString("FileInterne")))
			xmlReader.readNext();
		if (!xmlReader.atEnd())
			calibXml = xmlReader.readElementText();

			//conversion du fichier de calibration
		if (!calibXml.isEmpty()) {
			calibXml2 = calibXml.section(".",0,-2) + QString("2.xml");
			if (!QFile(calibXml2).exists()) {
				QFile fxml(calibXml);
				if (!fxml.open(QIODevice::ReadOnly | QFile::Text))
					return conv(tr("Fail to read file %1.")).arg(calibXml);
				QTextStream inStream(&fxml);

				QFile fxml2(calibXml2);
				if (!fxml2.open(QIODevice::WriteOnly | QIODevice::Truncate))
					return conv(tr("Fail to create file %1.")).arg(calibXml2);
				QTextStream outStream(&fxml2);

				QString text = inStream.readAll();
				text.replace(".",",");		
				outStream << text;
				fxml.close();
				fxml2.close();
			}
		}

			//conversion du fichier des poses
		QString orient = file.section(".",0,-2) + QString("2.xml");
		if (!QFile(orient).exists()) {
			QFile oldFile(file);
			if (!oldFile.open(QIODevice::ReadOnly | QIODevice::Text))
				return conv(tr("Fail to read file %1.")).arg(file);
			QTextStream inStream(&oldFile);
			QFile newFile(orient);
			if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
				return conv(tr("Fail to create file %1.")).arg(orient);
			QTextStream outStream(&newFile);
			QString text = inStream.readAll();
			text.replace(".",",");	
			text.replace(",xml",".xml");	//cas particulier des fichiers inclus
			if (!calibXml.isEmpty()) text.replace(calibXml,calibXml2);
			outStream << text;
			oldFile.close();
			newFile.close();
		}

		//lecture
		QString fichier = (virgule==QString("."))? file : orient;
		CamStenope* cam = NS_ParamChantierPhotogram::Cam_Gen_From_File(fichier.toStdString(), string("OrientationConique"), 0)->CS();	//ElCamera::CS = static_cast<CamStenope*>(ElCamera) (pas de delete ElCamera)
		if (cam==0) return conv(tr("Unvalid camera %1.")).arg(file);
		cameras[i].setCamera(cam);
	}
	return QString();
}

bool VueChantier::addNuages(const QVector<QString>& imgRef, const ParamMain& paramMain) {
	//ajout des outils
	nuagesLayers->create(tr("Point clouds"), imgRef);
	connect(nuagesLayers, SIGNAL(stateChanged(int,bool)), glWidget, SLOT(dispNuagLayers(int,bool)));

	//ajout des nuages de points
	aperoLayers->setChecked(1,false);
        if (!glparams.addNuages(paramMain)) return false;
	emit dispNuages();
	return true;
}
GLParams& VueChantier::modifParams () { return glparams; }

const QStringList VueChantier::getRefImg () const { return refBox->getSelectedCam(); }
const QVector<Pose>& VueChantier::getPoses () const { return glparams.getPoses(); }
const GLParams& VueChantier::getParams () const { return glparams; }
bool VueChantier::isDone() const { return done; }

///////////////////////////////////////////////////////////////////////////////////////////////


RotationButton::RotationButton() : QToolButton() {
	setFocusPolicy(Qt::ClickFocus);;
	icone = new QIcon(g_iconDirectory+"rotation.png");
	setIcon(*icone);
	setToolTip(conv(tr("Rotations (clic on corresponding circle)")));
	setFixedSize(QSize(100,100));
	setIconSize(QSize(100,100));
}
RotationButton::~RotationButton() { delete icone; }

int RotationButton::getRotation() { return rotation; }

void RotationButton::mousePressEvent(QMouseEvent * event) {
	rot0 = false;
	lastPos = QPoint(event->pos());
	pressPos = QPoint(event->pos());
	if (pressPos.x()<double(width())*3.0/8.0) {
		if (pressPos.y()<double(height())*3.0/8.0)
			rotation = 6;
		else if (pressPos.y()<double(height())*5.0/8.0)
			rotation = 3;
		else
			rotation = 6;
	} else if (pressPos.x()<double(width())*5.0/8.0) {
		if (pressPos.y()<double(height())*3.0/8.0)
			rotation = 1;
		else if (pressPos.y()<double(height())*5.0/8.0) {
			rotation = 0;
			rot0 = true;
		} else
			rotation = 2;
	} else {
		if (pressPos.y()<double(height())*3.0/8.0)
			rotation = 5;
		else if (pressPos.y()<double(height())*5.0/8.0)
			rotation = 4;
		else
			rotation = 5;
	}
}
void RotationButton::mouseMoveEvent(QMouseEvent * event) {
	if (event->x()<0 || event->y()<0 || event->x()>width() || event->y()>height())
		return;
	if (rot0 && rotation!=2) {
		rotation = 2;
		mouseMoveEvent(event);
		rotation = 4;
	}
	double angle;
	double angl1,angl2;
	switch (rotation) {
		case 1 : 
		case 2 : 
			rotation = 2;
			angle = (asin(double(event->y())*2.0/double(height())-1) - asin(double(lastPos.y())*2.0/double(height())-1)) * 180.0/PI;
			break;
		case 3 : 
		case 4 : 
			rotation = 4;
			angle = (asin(double(event->x())*2.0/double(width())-1) - asin(double(lastPos.x())*2.0/double(width())-1)) * 180.0/PI;
			break;
		case 5 : 
		case 6 : 
			rotation = 6;
			angl1 = atan2(double(lastPos.y())*2.0/double(height())-1,double(lastPos.x())*2.0/double(width())-1);
			angl2 = atan2(double(event->y())*2.0/double(height())-1,double(event->x())*2.0/double(width())-1);
			if (angl1-angl2>PI)
				angle = (angl2 - angl1 + 2*PI) * 180.0/PI;
			else if (angl1-angl2<-PI)
				angle = (angl2 - angl1 - 2*PI) * 180.0/PI;
			else
				angle = (angl1 - angl2) * 180.0/PI;
			break;
		default : return;
	}	
	if (!rot0 || rotation!=2) {
		lastPos = event->pos();
	}
	emit roll(rotation,angle);
}
void RotationButton::mouseReleaseEvent(QMouseEvent * event) {
	if (abs(event->x()-pressPos.x())<2 && abs(event->y()-pressPos.y())<2) {		//la souris n'a (presque) pas bougé
		emit roll(rotation,10.0);		//rotation d'angle fixé
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////


Pose::Pose():
	nomImg(QString()),
	camera(0),
	ptsAppui(QList<pair<Pt3dr, QColor> >()),
	ptsAppui2nd(QList<pair<Pt3dr, QColor> >()),
	emprise(QVector<Pt3dr>(4)),
	imgSize(QSize(0,0)) {}
Pose::Pose(const Pose& pose) { copie(pose); }
Pose::~Pose() {
	if (camera!=0)
		delete camera;
}

Pose& Pose::operator=(const Pose& pose) {
	if (&pose!=this)
		copie(pose);
        return *this;
}
void Pose::copie(const Pose& pose) {
	nomImg = pose.getNomImg();
	camera = &pose.getCamera();
	ptsAppui = pose.getPtsAppui();
	ptsAppui2nd = pose.getPtsAppui2nd();
	emprise = pose.getEmprise();
	imgSize = QSize(pose.width(),pose.height());
}

void Pose::setNomImg (const QString& img) { nomImg = img; }
void Pose::setCamera(CamStenope* cam) { camera = cam; }	//! au delete
void Pose::setEmprise(const QVector<Pt3dr>& rec) { emprise = rec; }
void Pose::setImgSize(const QSize& s) { imgSize = s; }
QList<pair<Pt3dr, QColor> >& Pose::modifPtsAppui() { return ptsAppui; }	
QList<pair<Pt3dr, QColor> >& Pose::modifPtsAppui2nd() { return ptsAppui2nd; }	

const QString& Pose::getNomImg () const { return nomImg; }
const CamStenope& Pose::getCamera() const { return *camera; }
const QList<pair<Pt3dr, QColor> >& Pose::getPtsAppui() const { return ptsAppui; }	
const QList<pair<Pt3dr, QColor> >& Pose::getPtsAppui2nd() const { return ptsAppui2nd; }	
const QVector<Pt3dr>& Pose::getEmprise() const { return emprise; }
Pt3dr Pose::centre() const { return camera->VraiOpticalCenter(); }
const ElMatrix<REAL>& Pose::rotation() const { return camera->Orient().Mat(); }
int Pose::width() const { return imgSize.width(); }
int Pose::height() const { return imgSize.height(); }

QVector<REAL> Pose::centre2() const {
	QVector<REAL> C(3,0);
	C[0] = centre().x;
	C[1] = centre().y;
	C[2] = centre().z;
	return C;
}
QVector<REAL> Pose::direction() const {
	ElMatrix<REAL> R = rotation();
	QVector<REAL> D(3,0);
	for (int i=0; i<3; i++) D[i] = R(i,2);
	REAL norm = 0;
        for (int i=0; i<3; i++) norm += D.at(i) * D.at(i);
	norm = sqrt(norm);
	for (int i=0; i<3; i++) D[i] /= norm;
	return D;
}
void Pose::simplifie2nd() {
	QList<pair<Pt3dr, QColor> >::iterator it=ptsAppui2nd.begin();
	while (it!=ptsAppui2nd.end()) {
		if (ptsAppui.contains(*it))
			ptsAppui2nd.erase(it);
		else
			it++;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////


Layers::Layers() : QGroupBox(), ckeckBoxes(0) {}
Layers::~Layers() {
	if (ckeckBoxes!=0) {
		for (int i=0; i<count; i++)
			delete ckeckBoxes[i];	
		delete [] ckeckBoxes;
	}
}

void Layers::create(QString title, const QVector<QString>& noms)
{
	count = noms.count();
	setTitle(title);
	layout = new QVBoxLayout;
	ckeckBoxes = new QCheckBox* [count];
	QSignalMapper* mapper = new QSignalMapper; 	
	for (int i=0; i<count; i++) {
		ckeckBoxes[i] = new QCheckBox(noms.at(i));
		ckeckBoxes[i]->setCheckable(true);
		ckeckBoxes[i]->setChecked(true);
		layout->addWidget(ckeckBoxes[i]);
		connect(ckeckBoxes[i], SIGNAL(stateChanged(int)), mapper, SLOT(map()));
		mapper->setMapping(ckeckBoxes[i], i);
	}
	setLayout(layout);
	connect(mapper, SIGNAL(mapped(int)),this, SLOT(emitChanges(int)));
}

void Layers::emitChanges(int layer) {
	emit (stateChanged(layer, ckeckBoxes[layer]->isChecked()));
}

void Layers::setChecked(int layer, bool checked) {
	ckeckBoxes[layer]->setChecked(checked);
}

void Layers::addWidget(QWidget* widget, Qt::Alignment alignment) {
	layout->addWidget(widget, 0, alignment);
}


///////////////////////////////////////////////////////////////////////////////////////////////


GLuint GLParams::nbGLLists = 0;
double GLParams::maxScale = 500;
GLParams::GLParams() : dossier(QString()), poses(QVector<Pose>()), distance(0), zoneChantier(QVector<GLdouble>(6,0)), zoneChantierEtCam(QVector<GLdouble>(6,0)), rot(0), trans(QVector<GLdouble>(3,0)), scale(1), focale(500), camlayers(QVector<bool>(3,true)), nuaglayers(QVector<bool>()), zoneNuage(QVector<REAL>(6,0)), color(Texture), measure(Aucune), sgt(pair<Pt3dr,Pt3dr>()) {
	camlayers[2] = false;
}
GLParams::GLParams(const GLParams& params) { copie(params); }
GLParams::~GLParams() {
	if (rot!=0)
		delete [] rot;
}

GLParams& GLParams::operator=(const GLParams& params) {
	if (&params!=this)
		copie(params);
	return *this;
}
void GLParams::copie(const GLParams& params) {
        dossier = params.getDossier();
        poses = params.getPoses();
	distance = params.getDistance();
        zoneChantier = params.getZoneChantier();
        zoneChantierEtCam = params.getZoneChantierEtCam();
        if (params.getRot()==0) rot = 0;
	else {
		for (int i=0; i<16; i++)
                        rot[i] = params.getRot()[i];
	}
	trans = params.getTrans();
	scale = params.getScale();
	focale = params.getFocale();
	camlayers = params.getCamlayers();
	nuaglayers = params.getNuaglayers();
	nuages = params.getNuages();
	zoneNuage = params.getZonenuage();
	color = params.getColor();
	measure = params.getMesure();
	sgt = params.getSegment();
}

bool GLParams::addNuages(const ParamMain& paramMain) {
	//limite de chaque nuage
	for (int i=0; i<3; i++) {
		zoneNuage[2*i] = numeric_limits<REAL>::max();
		zoneNuage[2*i+1] = -numeric_limits<REAL>::max();
	}

	//limite de l'ensemble des nuages
        for (int i=0; i<nuages.count(); i++) {
		QVector<REAL> zone(6);
		for (int j=0; j<3; j++) {
			zone[2*j] = numeric_limits<REAL>::max();
			zone[2*j+1] = -numeric_limits<REAL>::max();
		}
                cElNuage3DMaille* nuage = nuages.at(i).getPoints().at(5);
		cElNuage3DMaille::tIndex2D it=nuage->Begin();
		while(it!=nuage->End()) {
			Pt3dr P = nuage->PtOfIndex(it);
                        if (P.x<zone.at(0)) zone[0] = P.x;
                        if (P.x>zone.at(1)) zone[1] = P.x;
                        if (P.y<zone.at(2)) zone[2] = P.y;
                        if (P.y>zone.at(3)) zone[3] = P.y;
                        if (P.z<zone.at(4)) zone[4] = P.z;
                        if (P.z>zone.at(5)) zone[5] = P.z;
			nuage->IncrIndex(it);			
		}
                nuages[i].setZone(zone);
		for (int j=0; j<3; j++) {
                        if (zone.at(2*j)<zoneNuage.at(2*j)) zoneNuage[2*j] = zone.at(2*j);
                        if (zone.at(2*j+1)>zoneNuage.at(2*j+1)) zoneNuage[2*j+1] = zone.at(2*j+1);
		}
	}

	//association des poses aux nuages
        for (int i=0; i<nuages.count(); i++) {
		bool ok;
		QString numCarte = paramMain.getNumImage( nuages.at(i).getCarte().section("/",-1,-1), &ok, false );
		if (!ok) {
			cout << conv(QObject::tr("Fail to extract image %1 number.")).arg(nuages.at(i).getCarte().section("/",-1,-1)).toStdString() << endl;
			return false;
		}
                for (int j=0; j<poses.count(); j++) {
			QString numPose = paramMain.getNumImage( poses.at(j).getNomImg(), &ok, false );
			if (!ok) {
				cout << conv(QObject::tr("Fail to extract image %1 number.")).arg(poses.at(j).getNomImg()).toStdString() << endl;
				return false;
			}
			if (numCarte==numPose) {
			    nuages[i].setPose( poses[j] );
				break;
			}
		}
	}
	return true;
}

const QString& GLParams::getDossier() const { return dossier; }
const QVector<Pose>& GLParams::getPoses() const { return poses; }
double GLParams::getDistance() const { return distance; }
const QVector<GLdouble>& GLParams::getZoneChantier() const { return zoneChantier; }
const QVector<GLdouble>& GLParams::getZoneChantierEtCam() const { return zoneChantierEtCam; }
const GLdouble* GLParams::getRot() const { return rot; }
const QVector<GLdouble>& GLParams::getTrans() const { return trans; }
const GLdouble& GLParams::getScale() const { return scale; }
int GLParams::getFocale() const { return focale; }
const QVector<bool>& GLParams::getCamlayers() const { return camlayers; }
const QVector<bool>& GLParams::getNuaglayers() const { return nuaglayers; }
const QVector<Nuage>& GLParams::getNuages() const { return nuages; }
const QVector<REAL>& GLParams::getZonenuage() const { return zoneNuage; }
const GLParams::Couleur& GLParams::getColor() const { return color; }
const GLParams::Mesure& GLParams::getMesure() const { return measure; }
const std::pair<Pt3dr,Pt3dr>& GLParams::getSegment() const { return sgt; }
GLuint GLParams::getNbGLLists() const { return nbGLLists; }
double GLParams::getMaxScale() const { return maxScale; }

void GLParams::setDossier(const QString& dir) { dossier = dir; }
QVector<Pose>& GLParams::modifPoses() { return poses; }
void GLParams::setDistance(double echelle) { distance = echelle; }
void GLParams::setZoneChantier(const QVector<GLdouble>& zc) { zoneChantier = zc; }
void GLParams::setZoneChantierEtCam(const QVector<GLdouble>& zcec) { zoneChantierEtCam = zcec; }
GLdouble*& GLParams::modifRot() { return rot; }
QVector<GLdouble>& GLParams::modifTrans() { return trans; }
void GLParams::setScale(const GLdouble& sc) { scale = sc; }
void GLParams::setFocale(int f) { focale = f; }
QVector<bool>& GLParams::modifCamlayers() { return camlayers; }
QVector<bool>& GLParams::modifNuaglayers() { return nuaglayers; }
QVector<Nuage>& GLParams::modifNuages() { return nuages; }
QVector<REAL>& GLParams::modifZonenuage() { return zoneNuage; }
void GLParams::setColor(const Couleur& c) { color = c; }
void GLParams::setMesure(const Mesure& m) { measure = m; }
void GLParams::setSegment(const std::pair<Pt3dr,Pt3dr>& p) { sgt = p; }
void GLParams::incrNbGLLists() { nbGLLists++; }
void GLParams::resetNbGLLists() { nbGLLists = 0; }

///////////////////////////////////////////////////////////////////////////////////////////////

		
int Nuage::nbResol = 6;
Nuage::Nuage() : points(QVector<cElNuage3DMaille*>(nbResol,0)), correlation(QVector<QString>(nbResol)), carte(QString()), imageCouleur(QString()), pose(0), focale(0), zone(QVector<REAL>(8,0)), fromTA(false), georefMNT(GeorefMNT()), zoomMax(nbResol-1) {}
Nuage::Nuage(const Nuage& nuage) { copie(nuage); }
Nuage::~Nuage() {
	for (int i=0; i<points.count(); i++) {
		   if (points.at(i)!=0)
			 delete points.at(i);
	}
}

Nuage& Nuage::operator=(const Nuage& nuage) {
	if (&nuage!=this)
		copie(nuage);
	return *this;
}
void Nuage::copie(const Nuage& nuage) {
	points = nuage.getPoints();
	correlation = nuage.getCorrelation();
	carte = nuage.getCarte();
	imageCouleur = nuage.getImageCouleur();
        *pose = nuage.getPose();
	focale = nuage.getFocale();
	zone = nuage.getZone();
	fromTA = nuage.getFromTA();
	georefMNT = nuage.getGeorefMNT();
	zoomMax = nuage.getZoomMax();
}
		
const QVector<cElNuage3DMaille*>& Nuage::getPoints() const { return points; }
const QVector<QString>& Nuage::getCorrelation() const { return correlation; }
const QString& Nuage::getCarte() const { return carte; }
const QString& Nuage::getImageCouleur() const { return imageCouleur; }
const Pose& Nuage::getPose() const { return *pose; }
int Nuage::getFocale() const { return focale; }
const QVector<REAL>& Nuage::getZone() const { return zone; }
int Nuage::getNbResol() const { return nbResol; }
bool Nuage::getFromTA() const { return fromTA; }
const GeorefMNT& Nuage::getGeorefMNT() const { return georefMNT; }
int Nuage::getZoomMax() const { return zoomMax; }

QVector<cElNuage3DMaille*>& Nuage::modifPoints() { return points; }
QVector<QString>& Nuage::modifCorrelation() { return correlation; }
void Nuage::setCarte(const QString& c) { carte = c; }
void Nuage::setImageCouleur(const QString& ic) { imageCouleur = ic; }
void Nuage::setPose(Pose& p) { pose = &p; }
void Nuage::setFocale(int f) { focale = f; }
void Nuage::setZone(const QVector<REAL>& z) { zone = z; }
void Nuage::setFromTA(bool b) { fromTA = b; }
void Nuage::setGeorefMNT(const GeorefMNT& grm) { georefMNT = grm; }
void Nuage::setZoomMax(int z) { zoomMax = z; }

///////////////////////////////////////////////////////////////////////////////////////////////

		
SelectCamBox::SelectCamBox():
	QGroupBox(),
	mainLayout(0),
	refButton(0),
	okButton(0),
	camList(0),
	cutAct(0),
	camEdit(0),
	refImage(QString()),
	pCurrentMode(SelectCamBox::Hide) {}
SelectCamBox::~SelectCamBox() {
	clearContent();
}

void SelectCamBox::create(const Mode& mode, const QString& refImg, const QStringList& precCam) {	//Hide ou RefImage
	if (refButton!=0) refButton->setChecked(false);
	if (okButton!=0) okButton->setEnabled(false);
	if (camEdit!=0) camEdit->clear();
	if (camList!=0) camList->clear();

	clearContent();
	pCurrentMode = mode;
	if (pCurrentMode==Hide) return;
	if (pCurrentMode==RefImage) refImage = QString();
	else refImage = refImg;
	
	if (pCurrentMode==RefImage) setTitle(conv(tr("Reference image selection")));
	else setTitle(conv(tr("Camera selection for correlation")));

	refButton = new QToolButton ;
	refButton->setIcon(QIcon(g_iconDirectory+"camera.png"));
	refButton->setToolTip(conv(tr("Select a camera")));
	refButton->setCheckable(true);
	refButton->setChecked(false);

	okButton = new QToolButton;
	okButton->setIcon(QIcon(g_iconDirectory+"linguist-check-on.png"));
	okButton->setToolTip(tr("Accept"));
	okButton->setEnabled(false);

	if (pCurrentMode==RefImage) {
		camList = 0;
		cutAct = 0;
		camEdit = new QLineEdit;
		camEdit->setEnabled(false);
	} else {
		camEdit = 0;
		camList = new QListWidget;
		cutAct = new QAction(tr("&Remove"), this);
		connect(cutAct, SIGNAL(triggered()), this, SLOT(cut()));
		camList->setSelectionMode (QAbstractItemView::ExtendedSelection);
		if (precCam.count()>0) {
			for (int i=0; i<precCam.count(); i++) {
				if (camList->findItems(precCam.at(i),Qt::MatchExactly).count()>0) continue;
				QListWidgetItem* lwi = new QListWidgetItem(precCam.at(i));
				camList->addItem(lwi);
			}
			okButton->setEnabled(true);
		}
	}

	mainLayout = new QHBoxLayout;
	mainLayout->addWidget(refButton);
	mainLayout->addWidget(okButton);
	if (pCurrentMode==RefImage) mainLayout->addWidget(camEdit);
	else mainLayout->addWidget(camList);
	setLayout(mainLayout);

	connect(refButton, SIGNAL(clicked()), this, SIGNAL(refClicked()));
	connect(okButton, SIGNAL(clicked()), this, SIGNAL(okClicked()));
}

void SelectCamBox::addCamera(const QString& camera) {
	if (pCurrentMode==RefImage)
		camEdit->setText(camera);
	else if (pCurrentMode==CorrelImages) {
		if (camList->findItems(camera,Qt::MatchExactly).count()>0) return;
		if (camera==refImage) return;
		QListWidgetItem* lwi = new QListWidgetItem(camera);
		camList->addItem(lwi);
	}
	okButton->setEnabled(true);
}

void SelectCamBox::setRefButtonChecked(bool b) {
	if (refButton!=0)
		refButton->setChecked(b);
}

QStringList SelectCamBox::getSelectedCam() const {
	QStringList l ;
	if (pCurrentMode==RefImage)
		l.push_back(camEdit->text());
	else if (pCurrentMode==CorrelImages) {
		for (int i=0; i<camList->count(); i++)
			l.push_back(camList->item(i)->text());
	}
	return l;	
}

const SelectCamBox::Mode& SelectCamBox::getMode() const { return pCurrentMode; }

bool SelectCamBox::isRefButtonChecked() const {
	if (refButton!=0) return refButton->isChecked();
	else return false;
}

void SelectCamBox::clearContent() {
	setTitle(QString());
	if(mainLayout!=0) {
		delete mainLayout;
		mainLayout = 0;
	}
	if(refButton!=0) {
		delete refButton;
		refButton = 0;
	}
	if(okButton!=0) {
		delete okButton;
		okButton = 0;
	}
	if(camList!=0) {
		delete camList;
		camList = 0;
	}
	if(camEdit!=0) {
		delete camEdit;
		camEdit = 0;
	}
	if(cutAct!=0) {
		delete cutAct;
		cutAct = 0;
	}
}

void SelectCamBox::contextMenuEvent(QContextMenuEvent *event) {
	if (camList!=0 && camList->geometry().contains(event->pos()) && camList->selectedItems().size()!=0) {
		QMenu menu(this);
		menu.addAction(cutAct);
		menu.exec(event->globalPos());
	}
}
void SelectCamBox::cut() {
	//supprime la caméra sélectionnée de la liste
	for (int i=0; i<camList->count(); i++) {
		if (camList->item(i)->isSelected()) {
			QListWidgetItem* lwi = camList->takeItem(i);
			delete lwi;
			i--;
		}
	}
	if (camList->count()==0)
		okButton->setEnabled(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


GeorefMNT::GeorefMNT() : x0(0), y0(0), dx(1), dy(1), fichier(QString()), geomTerrain(false), done(true) {}
GeorefMNT::GeorefMNT(const GeorefMNT& georefMNT) { copie(georefMNT); }
GeorefMNT::GeorefMNT(const QString& f, bool lire):
	x0(0),
	y0(0),
	dx(1),
	dy(1),
	fichier(f),
	geomTerrain(false),
	done(false)
{
	if (lire) {
		geomTerrain = true;
		QString err = FichierGeorefMNT::lire(*this);
		if (!err.isEmpty()) {
			done = true;
			cout << ch(err) << endl;
		}
	} else
		done = true;
}
GeorefMNT::~GeorefMNT() {}

GeorefMNT& GeorefMNT::operator=(const GeorefMNT& cgeorefMNT) {
	if (this!=&cgeorefMNT) copie(cgeorefMNT);
	return *this;
}

void GeorefMNT::copie(const GeorefMNT& georefMNT) {
	x0 = georefMNT.getX0();
	y0 = georefMNT.getY0();
	dx = georefMNT.getDx();
	dy = georefMNT.getDy();
	fichier = georefMNT.getFichier();
	geomTerrain = georefMNT.getGeomTerrain();
}

Pt2dr GeorefMNT::terrain2Image(const Pt2dr& P) const {
	if (!geomTerrain) return P;
	REAL x = (P.x-x0)/dx;
	REAL y = (P.y-y0)/dy;
	return Pt2dr(x,y);
}

bool GeorefMNT::getDone() const { return done; }
double GeorefMNT::getX0() const { return x0; }
double GeorefMNT::getY0() const { return y0; }
double GeorefMNT::getDx() const { return dx; }
double GeorefMNT::getDy() const { return dy; }
const QString& GeorefMNT::getFichier() const { return fichier; }
bool GeorefMNT::getGeomTerrain() const { return geomTerrain; }

void GeorefMNT::setX0(double x) { x0 = x; }
void GeorefMNT::setY0(double y) { y0 = y; }
void GeorefMNT::setDx(double d) { dx = d; }
void GeorefMNT::setDy(double d) { dy = d; }
void GeorefMNT::setFichier(const QString& f) { fichier = f; }
void GeorefMNT::setGeomTerrain(bool g) { geomTerrain = g; }
