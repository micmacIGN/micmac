/*
interfMicmac.h et interfMicmac.cpp correspondent au menu Calcul -> cartes de profondeur qui permet de calculer des cartes de profondeur à partir d'une image maîtresse, d'un masque et d'images pour la corrélation.
La classe interfMicmac correspond à cette fenêtre et la classe MaskTab correspond à l'(unique) onglet principal :
* resizableWidget permet de placer un ascenseur dans l'onglet et évite ainsi que la fenêtre dépasse de l'écran ;
* le QTreeWidget treeWidget liste les cartes de pronfondeur dont les paramètres ont été saisis et à calculer ;
* la QListWidget correlImgsList permet de lister les images pour la corrélation ;
* si la carte à calculer doit être dans le repère terrain, le bouton TAButton permet de lancer le calcul du tableau d'assemblage (par MicMAc) et utilise ce TA comme image de référence por le dessin du masque ;
La classe MasqueWidget est commune avec la partie saisie du plan horizontal pour déinir l'orientation absolue dans interfApero.h. Elle regroupe les widgets de saisie des paramètres (image de référence, saisie, modification et enregistrement du masque) et intègre les widgets supplémentaire de interfMicmac (TA, images pour la corrélation).
La classe ImgCorrelChoice, quand elle est ajoutée dans la QListWidget correlImgsList, permet de sélectionner l'image pour la corrélation et de valider ce choix pour l'ajouter dans la liste.
La classe CarteDeProfondeur regroupe les paramètres d'une carte de profondeur.
La classe ParamMasqueXml regroupe les paramètres d'un masque afin d'écrire son fichier de référencement dès sauvegarde des paramètres de la carte.
*/

#ifndef INTERFMICMAC_H
#define INTERFMICMAC_H

#include "all.h"

/* fenêtre de saisie des paramètres pour MicMac et d'ouverture de la fenêtre de dessin/modification du masque */

QString imgNontuilee(const QString& img);

class InterfMicmac;

class CarteDeProfondeur
{
	public :
		CarteDeProfondeur();
		CarteDeProfondeur(const CarteDeProfondeur& carteDeProfondeur);
		~CarteDeProfondeur();

		CarteDeProfondeur& operator=(const CarteDeProfondeur& carteDeProfondeur);
		bool ecrireCorrelListe(const ParamMain& paramMain) const;
		QString getMasque(const ParamMain& paramMain) const;	//avec le dossier
		QString getRepereFile(const ParamMain& paramMain) const;	//sans le dossier
		QString getReferencementMasque(const ParamMain& paramMain) const;	//avec le dossier
		QString getImageSaisie(const ParamMain& paramMain) const;	//avec le dossier	//image utilisée pour la saisie du masque (imageDeReference si repère image, TA si repère terrain) ; par défaut le masque s'appellera masque_idxImageSaisie.tif

		bool getACalculer() const;
		const QString& getImageDeReference() const;	//sans le dossier
		const QStringList& getImagesCorrel() const;
		QStringList& modifImagesCorrel();
		bool getRepere() const;
		bool getAutreRepere() const;
		const QString& getImgRepMasq() const;
		const QString& getImgRep() const;
		const std::pair<QPoint,QPoint>& getSegmentRep() const;
		const QPoint& getAxeRep() const;
		bool getDoOrtho() const;
		bool getOrthoCalculee() const;
		const QStringList& getImgsOrtho() const;
		QStringList& modifImgsOrtho();
		double getEchelleOrtho() const;
		const std::pair<QString,QString>& getImgEchOrtho() const;
		const QVector<QPoint>& getPtsEchOrtho() const;
		const std::pair<float,float>& getInterv() const;
		bool getDiscontinuites() const;
		float getSeuilZ() const;
		float getSeuilZRelatif() const;

		void setACalculer(bool a);
		void setImageDeReference(const QString& i);
		void setImagesCorrel(const QStringList& im);
		void setRepere(bool r);
		void setAutreRepere(bool n);
		void setImgRepMasq(const QString& img);
		void setImgRep(const QString& imgR);
		void setSegmentRep(const std::pair<QPoint,QPoint>& s);
		void setAxeRep(const QPoint& a);
		void setDoOrtho(bool d);
		void setOrthoCalculee(bool o);
		void setImgsOrtho(const QStringList& imgs);
		void setEchelleOrtho(double e);
		void setImgEchOrtho(const std::pair<QString,QString>& imgE);
		void setPtsEchOrtho(const QVector<QPoint>& p);
		void setInterv(const std::pair<float,float>& in);
		void setDiscontinuites(bool di);
		void setSeuilZ(float se);
		void setSeuilZRelatif(float seu);

	private :
		void copie(const CarteDeProfondeur& carteDeProfondeur);

		//MNT
		bool aCalculer;
		QString imageDeReference;
		QStringList imagesCorrel;	//images pour la corrélation
		//repère
		bool repere;	//true si repère image
		bool autreRepere;	//true si repère euclidien autre que le repère de l'aéro : par défaut le repère s'appellera repere_idxImageDeReference.xml
			//pour la saisie
		QString imgRepMasq;	//par défaut le masque s'appellera imgMasqPlan.section(".",0,-2)+QString("_MasqRepTA.tif");
		QString imgRep; //img pour la direction
		std::pair<QPoint,QPoint> segmentRep; //direction
		QPoint axeRep;
		//orthos
		bool doOrtho;
		bool orthoCalculee;	//pour Porto
		QStringList imgsOrtho;
		double echelleOrtho;
			//saisie
		std::pair<QString,QString> imgEchOrtho;
		QVector<QPoint> ptsEchOrtho;
			//relief
		std::pair<float,float> interv;
		bool discontinuites;
		float seuilZ;
		float seuilZRelatif;		
};


class ListeWindow : public QDialog
{
	Q_OBJECT

	public :
		ListeWindow (QWidget* parent, const QStringList& images);
		~ListeWindow();

		QStringList getSelectedImages();

	private :
		QListWidget* liste;
};

class CartesTab : public QScrollArea
{
	Q_OBJECT

	public:
		CartesTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  QVector<CarteDeProfondeur>* param);
		~CartesTab();

		void enableSelect(bool b);
		void updateListe();
		int getCarte(const QVector<CarteDeProfondeur>& parametres, const QString& nomImg) const;

	signals:
		void modifCarte(CarteDeProfondeur*);
		void updateCalcButton();

	private slots:
		void selectionChanged();
		void removeCartes();
		void addNewCarte();
		void modifCarte();
		void ignore();

	private:
		QSize sizeHint();
		QSize minimumSizeHint();
		void resizeEvent(QResizeEvent*);
		void resizeTreeWidget();
		void contextMenuEvent(QContextMenuEvent *event);

		QTreeWidget *treeWidget;
		QPushButton *addButton;
		QPushButton *removeButton;
		QPushButton *modifButton;
		QAction 	*ignoreAct;
		QWidget		*resizableWidget;

		QVector<CarteDeProfondeur> *parametres;
		CarteDeProfondeur 			carteCourante;
		InterfMicmac			   *parent;
		const ParamMain			   *paramMain;
};


class MNTTab : public QScrollArea
{
	Q_OBJECT

	public:
		MNTTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  CarteDeProfondeur* param, VueChantier* vueChantier, const QVector<CarteDeProfondeur>* cartes, Assistant* help);
		~MNTTab();

	signals :
		void suiteMNT(bool);

	private slots:
		void imgRefChanged(QString img);
		void vue3DClicked();
		void addCorrelImgClicked();
		void addFromListClicked();
		void addFromViewClicked();
		void addFromStatClicked();
		void selectionChanged();
		void removeCorrelImgClicked();

	private:
		QSize sizeHint();
		void resizeEvent(QResizeEvent*);
		QSize minimumSizeHint();

		QComboBox	*imageCombo;
		QPushButton	*vue3DButton;
		QListWidget	*correlImgsList;
		QToolButton *addCorrelImgButton;
		QAction 	*addFromList;
		QAction 	*addFromView;
		QAction 	*addFromStat;
		QPushButton *removeCorrelImgButton;
		QWidget		*resizableWidget;

		Assistant			*assistant;
		CarteDeProfondeur	*parametre;
		InterfMicmac		*parent;
		const ParamMain		*paramMain;
		QString 			 dir;
		VueChantier			*vue3D;
};

class RepereTab : public QScrollArea
{
	Q_OBJECT

	public:
		RepereTab(InterfMicmac* interfMicmac, const ParamMain* pMain, VueChantier* vueChantier, CarteDeProfondeur* param, Assistant* help);
		~RepereTab();

	signals :
		void suiteRepere(bool);

	private slots:
		void repereClicked();
		void autreRepChecked();
		void updateParam(int);
		void importRepClicked();
		void openClicked();
		void vue3DClicked();
		void TAClicked();

	private:
		enum Mode {Begin, Image, Terrain, AutreRepere, NewRepere, OpenRepere, TACalcule};

		void updateInterface(RepereTab::Mode mode);
		QString QPoint2QString(const QPoint& P);
		QString calcTAName();

		QSignalMapper	*mapper;
		QRadioButton	*radioTerrain;
		QRadioButton	*radioImage;
		QCheckBox		*autreRepCheck;
		QGroupBox		*radio2Box;
		QRadioButton	*radioOpen;
		QRadioButton	*radioNew;
		QGroupBox		*openBox;
		QLineEdit		*openEdit;
		QPushButton		*openButton;
		QGroupBox		*paramBox;
		QPushButton		*vue3DButton;
		MasqueWidget	*masqueWidget;
		DirectionWidget	*directionWidget;
		QPushButton		*TAButton;
		QWidget			*resizableWidget;

		Assistant		  *assistant;
		CarteDeProfondeur *parametre;
		InterfMicmac	  *parent;
		const ParamMain	  *paramMain;
		VueChantier		  *vue3D;
		QString 		   dir;
};

class MaskTab : public QScrollArea
{
	Q_OBJECT

	public:
		MaskTab(InterfMicmac* interfMicmac, const ParamMain* pMain,  CarteDeProfondeur* param, Assistant* help);
		~MaskTab();

	signals :
		void suiteMasque();

	private slots:
		void updateParam();

	private:
		MasqueWidget *masqueWidget;
		QWidget		 *resizableWidget;

		Assistant		  *assistant;
		CarteDeProfondeur *parametre;
		InterfMicmac	  *parent;
		const ParamMain	  *paramMain;
		QString			   dir;
};

class OrthoTab : public QScrollArea
{
	Q_OBJECT

	public:
		OrthoTab(InterfMicmac* interfMicmac, const ParamMain* pMain, CarteDeProfondeur* param, Assistant* help);
		~OrthoTab();

	signals :
		void suiteOrtho();

	private slots:
		void orthoClicked();
		void selectionChanged();
		void addImgsClicked();
		void removeImgsClicked();
		void updateParam();

	private:
		QCheckBox		*checkOrtho;
		QGroupBox		*imgsBox;
		QListWidget		*listeWidget;
		QPushButton		*addImgsButton;
		QPushButton		*removeImgsButton;
		QGroupBox		*echelleBox;
		EchelleWidget	*echelleWidget;
		QWidget			*resizableWidget;

		Assistant			*assistant;
		InterfMicmac		*parent;
		const ParamMain		*paramMain;
		CarteDeProfondeur	*parametre;
		QString 			 dir;
};

class ProfondeurTab : public QScrollArea
{
	Q_OBJECT

	public:
		ProfondeurTab(InterfMicmac* interfMicmac, const ParamMain* pMain, CarteDeProfondeur* param);
		~ProfondeurTab();

	signals :
		void suiteProf();

	private slots:
		void discontClicked();
		void updateParam();

	private:
		QLineEdit* intervMinEdit;
		QLineEdit* intervMaxEdit;
		QCheckBox* checkDiscont;
		QGroupBox* discontBox;
		QLineEdit* regulAbsEdit;
		QLineEdit* regulEdit;
		QWidget* resizableWidget;

		InterfMicmac* parent;
		const ParamMain* paramMain;
		CarteDeProfondeur* parametre;
};

class ParamMasqueXml
{
	public :
		ParamMasqueXml ();
		ParamMasqueXml (
			const QString& mntFile,
			const QString& masqueFile,
			const QSize& nbPx,
			const QPointF& OrigXY = QPoint(0,0),
			const QPointF& ResolXY = QPoint(1,1),
			double OrigZ = 0,
			double ResolZ = 1,
			const QString& Geom = QString("eGeomMNTFaisceauIm1PrCh_Px1D")
		);
		ParamMasqueXml(const ParamMasqueXml& paramMasqueXml);
		~ParamMasqueXml();

		ParamMasqueXml& operator=(const ParamMasqueXml& paramMasqueXml);

		const QString& getNameFileMnt() const;
		const QString& getNameFileMasque() const;
		const QSize& getNombrePixels() const;
		const QPointF& getOriginePlani() const;
		const QPointF& getResolutionPlani() const;
		double getOrigineAlti() const;
		double getResolutionAlti() const;
		const QString& getGeometrie() const;

		void setNameFileMnt(const QString& N);
		void setNameFileMasque(const QString& Na);
		void setNombrePixels(const QSize& No);
		void setOriginePlani(const QPointF& O);
		void setResolutionPlani(const QPointF& R);
		void setOrigineAlti(double Or);
		void setResolutionAlti(double Re);
		void setGeometrie(const QString& G);		
		
	private :
		void copie(const ParamMasqueXml& paramMasqueXml);

		QString NameFileMnt;	//chemin absolu
		QString NameFileMasque;	//chemin absolu
		QSize NombrePixels;
		QPointF OriginePlani;
		QPointF ResolutionPlani;
		double OrigineAlti;
		double ResolutionAlti;
		QString Geometrie;
};

class InterfMicmac : public QDialog
{
	Q_OBJECT

	public:
		InterfMicmac(Interface* parent, const ParamMain* param, int typChan, VueChantier* vue, Assistant* help);
		~InterfMicmac();

		const QVector<CarteDeProfondeur>& getParamMicmac() const;
		void adjustSizeToContent();
		QSize maximumSizeHint() const;

	private slots:
		void modifCarte(CarteDeProfondeur* carte);
		void saveClicked();
		void precClicked();
		void suivClicked();
		void calcClicked();
		void helpClicked();
		void cancelClicked();
		void updateInterface();
		void suiteMNT(bool b);
		void suiteRepere(bool b);
		void suite();
		void updateCalcButton();

	private:
		bool cartesACalculer();

		QTabWidget	  *tabWidget;
		CartesTab	  *cartesTab;
		MNTTab		  *mNTTab;
		RepereTab	  *repereTab;
		MaskTab		  *maskTab;
		OrthoTab	  *orthoTab;
		ProfondeurTab *profondeurTab;
		QPushButton   *suivantButton;
		QPushButton   *precButton;
		QPushButton   *saveButton;
		QPushButton   *calButton;

		//données
		Assistant	    		   *assistant;
		const ParamMain 		   *paramMain;
		QVector<CarteDeProfondeur>  paramMicmac;
		int 						typeChantier;
		QString 					dir;
		CarteDeProfondeur		   *carteCourante;
		VueChantier				   *vueChantier;
};

#endif
