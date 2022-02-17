/* interfConversion.h et interfConversion.cpp correspondent au menu Conversion
La classe InterfCartes8B correspond à la fenêtre du menu Conversion -> Cartes de profondeur 8B ; elle permet de sélectionner les cartes à convertir ainsi que les paramètres de GrShade à appliquer pour toutes les cartes à convertir (couleurs hypsométriques, ombrage...).
La classe BoiteArea correspond à la saisie de la boîte englobante et est incluse dans InterfCartes8B.
La classe ParamConvert8B enregistre ces paramètres.
La classe ParamConvert8B::carte8B contient les paramètres d'une carte pouvant être convertie (résolution, image couleur...).
La classe InterModele3D correspond à la fenêtre associée au menu Conversion -> Modèles 3D ; elle permet de sélectionner les cartes à convertir au format ply ainsi que les paramètres de Nuage2Ply à appliquer (changement d'image de texture,répertoire desauvegarde).
La classe ParamNuages enregistre ces paramètres.
*/

#ifndef INTERFCONVERT8B_H
#define INTERFCONVERT8B_H

#include "all.h"


class ParamConvert8B
{
	public:
		class carte8B {
			public :
				carte8B();
				carte8B(const QString& c, int dz, QString nc, int e, const QString& m, const QString& ri, const QString& t=QString());
				carte8B(const carte8B& c);
				~carte8B();

				carte8B& operator=(const carte8B& c);

				const QString& getCarte16B() const;
				int getDezoom() const;
				QString getNumCarte() const;
				int getEtape() const;
				const QString& getMasque() const;
				const QString& getRefImg() const;
				const QString& getTexture() const;

			private :
				void copie(const carte8B& c);

				QString carte16B;
				int dezoom;
				QString numCarte;
				int etape;
				QString masque;
				QString refImg;
				QString texture;
		};

		ParamConvert8B();
		ParamConvert8B(const ParamConvert8B& paramConvert8B);
		~ParamConvert8B();

		ParamConvert8B& operator=(const ParamConvert8B& paramConvert8B);

		QList<ParamConvert8B::carte8B>& modifImages();
		const QList<ParamConvert8B::carte8B>& getImages() const;

		bool getCommande(const QString& micmacDir, const QString& dir, const carte8B& image, const QString& outimage, const QString& outstd, bool holdvisu=true);	//commande GrShade en fonction de l'image et des paramères actuels et exécution
		QString getOutFile(const QString& dossier, const carte8B& image);	//fichier de sortie en fonction de l'image et des paramères actuels
		const QList<ParamConvert8B::carte8B>& readImages() const;

		bool 				visualiser;
		bool 				useMasque;
		bool 				dequantifier;
		bool 				otherOptions;
		bool 				withImg;
		int 				nbDir;
		int 				modeOmbre;
		double 				fz;
		QString 			out;
		double  			anisotropie;
		double 				hypsoDyn;
		double 				hypsoSat;
		bool 				doBoite;
		pair<QPoint,QPoint> boite;	//x0,y0 , w,h
		int 				bord;

	private:
		void copie(const ParamConvert8B& paramConvert8B);

		QList<carte8B> images;
};

class BoiteArea : public QWidget
{
	Q_OBJECT

	public :
		BoiteArea();
		BoiteArea(const QString& img, int dz);
		~BoiteArea();

		pair<QPoint,QPoint> getBoite() const;	//retourne les coordonnées des points extrêmes
		void changeImage(const QString& img);

	signals :
		void changed();

	private :
		void mousePressEvent(QMouseEvent* event);
		void mouseMoveEvent(QMouseEvent* event);
		void mouseReleaseEvent(QMouseEvent*);
		void paintEvent(QPaintEvent*);
		QPoint P2() const;
		QPoint P4() const;
		double dist (const QPoint& P, const QPoint& PP) const;

		QImage image;
		QPoint P1;	//coord img réelle
		QPoint P3;
		int movingPoint;
		double dezoom;	//taille img 16b / taille img réelle
		double scale; //taille img réelle / taille img écran
		QPoint precPoint;
};

class InterfCartes8B : public QDialog
{
	Q_OBJECT

	public:
		InterfCartes8B(const QString& dossier, const QString& dossierMicmac, const QList<ParamConvert8B::carte8B>* cartes, QWidget* parent, Assistant* help);
		~InterfCartes8B();

		const ParamConvert8B& getParam() const;
		bool getDone() const;

	private slots:
		void showOptions();
		void optionChanged(int n=-1);
		void carteChanged();
		void calcClicked();
		void helpClicked();

	private:
		QString checkParametres(int i);

		QListWidget *listImages;
		QSignalMapper* mapper;	
		QCheckBox *checkVisu;
		QCheckBox *checkMask;
		QCheckBox *checkDequant;
		QPushButton* optionButton;
		QCheckBox* imageCheck;
		QSpinBox* nbdirBox;
		QComboBox* modeOmbreBox;
		QLineEdit* fzEdit;
		QLineEdit* outEdit;
		QLineEdit* anisoEdit;
		QLineEdit* hdynEdit;
		QLineEdit* hsatEdit;
		QCheckBox* boiteCheck;
		BoiteArea* boiteArea;
		QSpinBox* brdBox;
		QGroupBox* optionBox;
		QToolButton* apercuButton;
		QPushButton *calButton;
		QPushButton *cancelButton;
		QPushButton *helpButton;
		Assistant* assistant;

		//données
		const QList<ParamConvert8B::carte8B>* listeCartes;
		QString dir;	
		QString micmacDir;	
		ParamConvert8B paramConvert8B;
		bool done;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class InterfOrtho : public QDialog
{
	Q_OBJECT

	public:
		InterfOrtho(QWidget* parent, Assistant* help, ParamMain& param, QVector<CarteDeProfondeur>* cartesProfondeur);
		~InterfOrtho();

		bool getEgaliser() const;

	private slots:
		void calcClicked();
		void helpClicked();
		
	private:
		Assistant* assistant;
		QListWidget* listWidget;
		QCheckBox* checkEgal;
		QPushButton *calButton;
		QPushButton *cancelButton;
		QPushButton *helpButton;

		//données
		const ParamMain* paramMain;
		QVector<CarteDeProfondeur>* cartes;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ParamPly;
class ParamNuages
{
	public:
		ParamNuages();
		ParamNuages(const ParamNuages& paramNuages);
		~ParamNuages();

		ParamNuages& operator=(const ParamNuages& paramNuages);

		const CarteDeProfondeur& getParamMasque() const;
		const QString& getCarte() const;
		int getDezoom() const;
		QString getNumCarte() const;
		int getEtape() const;
		const QString& getFichierPly() const;
		const QString& getFichierXml() const;

		void setParamMasque(const CarteDeProfondeur* p);
		void setCarte(const QString& c);
		void setDezoom(int d);
		void setNumCarte(QString n);
		void setEtape(int i);
		void setFichierPly(const QString& s);

		void calcFileName(const QString& dossier);
		void updateFileName(const ParamPly& paramPly);
		QString commandeFiltrage(QString& masqueFiltre) const;
		QString commandePly(QString& commandeNuage2Ply, const QString& micmacDir, const ParamPly& paramPly, const QString& masqueFiltre, const ParamMain& paramMain) const;
		bool commandeMv() const;

	private:
		void copie(const ParamNuages& paramNuages);

		const CarteDeProfondeur* paramMasque;	//pointeur vers paramMain.getMasques()
		QString 				 carte;			//avec dossier
		int 					 dezoom;
		QString 				 numCarte;
		int 					 etape;
		QString					 fichierPly;	//avec dossier (conversion)
		QString					 fichierXml;	//avec dossier
};

class ParamPly
{
	public :
		ParamPly();
		/*ParamPly(int e, const QVector<bool>& n, bool d, bool doF, bool doE, const QList<pair<QString,QString> >& m, bool doB=false, const pair<QPoint,QPoint>& b=pair<QPoint,QPoint>(QPoint(0,0), QPoint(0,0)), bool b=true, bool doP=true, bool doX=false, double ex=1, double dy=1);*/
		ParamPly(int nbNuages);
		ParamPly(const ParamPly& paramPly);
		~ParamPly();

		ParamPly& operator=(const ParamPly& paramPly);

		int getEchantInterval() const;
		const QVector<bool>& getNuages() const;
		bool getDoFiltrage() const;
		bool getDoFusion() const;
		bool getDoEgalRadiom() const;
		const QList<pair<QString,QString> >& getMasques() const;
		bool getDoBoite() const;
		const pair<QPoint,QPoint>& getBoite() const;
		bool getBinaire() const;
		bool getDoPly() const;
		bool getDoXyz() const;
		double getExagZ() const;
		double getDyn() const;

		void setEchantInterval(int e);
		void setNuages(const QVector<bool>& n);
		QVector<bool>& modifNuages();
		void setDoFiltrage(bool d);
		void setDoFusion(bool doF);
		void setDoEgalRadiom(bool doE);
		void setMasques(const QList<pair<QString,QString> >& m);
		QList<pair<QString,QString> >& modifMasques();
		void setDoBoite(bool doB);
		void setBoite(const pair<QPoint,QPoint>& b);
		void setBinaire(bool b);
		void setDoPly(bool doP);
		void setDoXyz(bool doX);
		void setExagZ(double ex);
		void setDyn(double dy);

	private :
		void copie(const ParamPly& paramPly);

		int 							echantInterval;
		QVector<bool> 					nuages;
		bool 							doFiltrage;
		bool 							doFusion;
		bool							doEgalRadiom;
		QList<pair<QString,QString> >	masques;	//numCarte, masque
		bool 							doBoite;
		pair<QPoint,QPoint> 			boite;
		bool 							binaire;
		bool 							doPly;
		bool 							doXyz;
		double 							exagZ;
		double 							dyn;
};

class InterfModele3D : public QDialog
{
	Q_OBJECT

	public:
		InterfModele3D(QWidget* parent, Assistant* help, ParamMain& param, const QVector<ParamNuages>& choix);
		~InterfModele3D();

		const QVector<ParamNuages>& getModifications() const;
		const ParamPly& getParamPly() const;
		void chercheOrtho();	//si une orthoimage mosaïquée a été calculée, modifie cette fenêtre et les paramètres

	private slots:
		void optionChanged(int n=-1);
		void calcClicked();
		void helpClicked();
		void imgChoose();
		void dirPlyChoose();
		void drawMaskFiltr();
		void nuageChanged();

	private:
		void showEvent(QShowEvent*);
		void resizeTreeWidget();
		void contextMenuEvent(QContextMenuEvent *event);

		QTreeWidget		*treeWidget;
		QSpinBox		*intervBox;
		QCheckBox		*checkBinaire;
		QCheckBox		*checkPly;
		QCheckBox		*checkXyz;
		QLineEdit		*reliefEdit;
		QLineEdit		*dynEdit;
		QCheckBox		*boiteCheck;
		BoiteArea		*boiteArea;
		QCheckBox		*checkFiltr;
		QCheckBox		*checkFusion;
		QCheckBox		*checkRadiomEq;
		QLabel		  	*listeMasques;	//-> à rajouter pour montrer les masques modifiés
		QSignalMapper 	*mapper;	
		QPushButton		*calButton;
		QPushButton		*cancelButton;
		QPushButton		*helpButton;
		Assistant		*assistant;
		QAction			*changeImgAct;
		QAction			*savePlyAct;
		QAction			*maskFiltrPlyAct;
		QPoint 			 mouse;

		//données
		ParamMain			 *paramMain;
		QVector<ParamNuages>  nuages;
		ParamPly 			  paramPly;
		const QString		  dir;
};
#endif
