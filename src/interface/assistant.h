/* assistant.h et assistant.cpp regroupent les classes utiles au menu Aide de l'interface (le sous-menu A propos est une simple QMessageBox).
La classe Assistant permet d'afficher l'aide en ligne. Elle est appelée soit via le menu Aide -> Aide, soit via les boutons d'aide "?" répartis dans les différentes fenêtres de saisie de paramètres ; dans cas c'est la page concernant cette fenêtre qui est affichée.
La classe InterfOptions correspond au menu Aide -> Options ; cette fenêtre permet de modifier la langue de l'interface ou le dossier micmac. La classe Options enregistre ces paramètres.
La classe InterfVerifMicmac correspond au menu Aide -> Vérifier le calcul ; cette fenêtre, accessible uniquement pendant le calcul de Micmac, permet d'afficher un aperçu de la carte de profondeur en cours de calcul.
*/

#ifndef ASSISTANT
#define ASSISTANT

#include "all.h"


class Assistant
{
	public:
		Assistant();
		Assistant(const Assistant& assistant);
		~Assistant();

		void setPages(bool fr);
		void showDocumentation(const QString &file);
		Assistant& operator=(const Assistant& assistant);

		QString pageInterface;
		QString pageInterPastis;
		QString pageInterPastisChoice;
		QString pageInterPastisCouple;
		QString pageInterPastisCommun;
		QString pageInterfApero;
		QString pageInterfAperoMaitresse;
		QString pageInterfAperoReference;
		QString pageInterfAperoOrInit;
		QString pageInterfAperoAutocalib;
		QString pageInterfAperoMultiechelle;
		QString pageInterfAperoLibercalib;
		QString pageInterfAperoPtshomol;
		QString pageInterfMicmac;
		QString pageInterfMicmacMNT;
		QString pageInterfMicmacRepere;
		QString pageInterfMicmacMasque;
		QString pageInterfMicmacOrtho;
		QString pageInterfMicmacProfondeur;
		QString pageInterfCartes8B;
		QString pageInterfModeles3D;
		QString pageInterfOrtho;
		QString pageVueChantier;
		QString pageVueNuages;
		QString pageVueHomologues;
		QString pageDrawSegment;
		QString pageDrawPlanCorrel;
		QString pageInterfOptions;
		QString pageInterfVerifMicmac;
	    
	private:
		bool startAssistant();

		QProcess* proc;
};

/////////////////////////////////////////////////////////////////////////


//options et paramètres globaux de l'interface
class Options
{
	public:
		Options();
		Options(const Options& options);
		Options(const QSettings& settings);
		~Options();

		Options& operator=(const Options& options);

		const QString& getMicmacDir() const;
		const QLocale::Language& getLangue() const;
		int getCpu() const;
		const QList<std::pair<QString,double> >& getCameras() const;
		QList<std::pair<QString,double> >& modifCameras();

		void setMicmacDir(const QString& m);
		void setLangue(const QString& l);
		void setCpu(int c);

		bool updateSettings(QSettings& settings) const;
		static QStringList checkBinaries(const QString& micmacDossier);	//vérifie si les exécutables sont dans le dossier bin et renvoie tous les messages d'erreur
		static bool readMicMacInstall(QString& micmacDossier, int& cpuLu);
		static void writeMicMacInstall(const QString& micmacDossier, int cpuFinal);

	private:
		void copie(const Options& options);

		QString micmacDir;	//répertoire micmac
		QLocale::Language langue;	//langue
		int cpu;	//nombre maximal de processeurs à utiliser
		static QString micmacInstallFile;
		QList<std::pair<QString,double> > cameras;
};

class InterfOptions : public QDialog
{
	Q_OBJECT

	public:
		InterfOptions(QWidget* parent, Assistant* help, const QSettings& settings);
		~InterfOptions();

		const Options& getOptions() const;

	private slots:
		void micmacClicked();
		void langueClicked();
		void cpuClicked();
		void addCamClicked();
		void removeCamClicked();
		void okClicked();
		void helpClicked();

	private:
		QLineEdit* micmacEdit;
		QPushButton* micmacButton;
		QComboBox* langueCombo;
		QSpinBox* cpuSpin;
		QTreeWidget* camList;
		QPushButton *addCam;
		QPushButton *removeCam;
		QPushButton *okButton;
		QPushButton *cancelButton;
		QPushButton *helpButton;		

		//données
		Options options;
		Assistant* assistant;
		QLocale::Language oldlanguage;
		bool optChanged;
};

/////////////////////////////////////////////////////////////////////////

//vérification de la convergence du calcul de micmac
class ParamMain;
class InterfVerifMicmac : public QDialog
{
	Q_OBJECT

	public:
		InterfVerifMicmac(QWidget* parent, Assistant* help, const ParamMain* param, const QProgressDialog* pDialog);
		~InterfVerifMicmac();

	private slots:
		void majClicked();
		void helpClicked();

	private:	
		QPushButton* majButton;
		QToolButton* apercuButton;	//carte GrShade
		QToolButton* apercuButton2;	//image de corrélation
		QPushButton* helpButton;

		//données
		Assistant* assistant;
		const ParamMain* paramMain;
		const QProgressDialog* progressDialog;
};

/////////////////////////////////////////////////////////////////////////

//nettoyage des fichiers inutiles
class Nettoyeur
{
	public:
		Nettoyeur();
		Nettoyeur(const ParamMain* param);
		Nettoyeur(const Nettoyeur& nettoyeur);
		~Nettoyeur();

		Nettoyeur& operator=(const Nettoyeur& nettoyeur);
		void nettoie() const;

	private:
		friend void copie(Nettoyeur* nettoyeur1, const Nettoyeur& nettoyeur2);

		//données
		const ParamMain* paramMain;
};



#endif
