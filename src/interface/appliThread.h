/*
appliThread.h et appliThread.cpp regroupent les différentes classes permettant d'effectuer les calculs lancés par l'interface.
Chaque classe hérite de QThread et est donc lancée dans un thread indépendant du thread principal (affichage de l'interface et de la barre de progression), ce qui permet de mettre à jour la barre de progression sans interropre le calcul.
Le calcul est effectué par la fonction run(). Les paramètres saisis par l'utilisateur sont d'abord enregistrés dans différents fichiers xml (voir readwrite.h), puis les logiciels annexes (dossier micmac/bin) sont lancés via la fonction system. L'exécution des logiciels est indépendante de l'interface qui n'en a pas le contrôle ; l'avancement est récupéré par la classe Progression ; l'exécution est arrêtée (cas d'une erreur ou d'un arrêt par l'utilisateur) via la fonction killProcess.
Les différentes étapes sont enregistrées (setProgressLabel) pour pouvoir être affichées dans la barre de progression (avec la fonction Interface::updateProgressDialog, le thread de calcul ne peut pas modifier la barre de progression). Les erreurs sont aussi enregistrées (setReadError) pour affichage dans l'interface (Interface::displayErreur).
*/

#ifndef APPLITHREAD_H
#define APPLITHREAD_H

#include "all.h"

#define GLWidget GLWidget_IC

class AppliThread : public QThread
{
   Q_OBJECT

//regroupement des threads de calculs suivants (PastisThread, AperoThread, MicmacThread, Cartes8BThread)
	public:
		AppliThread();
		AppliThread(ParamMain* pMain, const QString& stdoutfilename, bool* annul);
		AppliThread(const AppliThread& appliThread);
		~AppliThread();

		AppliThread& operator=(const AppliThread& appliThread);
		
		virtual void run();
		virtual bool killProcess();

		int getEndResult() const;
		const QString& getProgressLabel() const;
		const QString& getReaderror() const;
		bool getIsDone() const;

	signals:
		void saveCalcul();

	protected :
		ParamMain* getParamMain();
		bool* getAnnulation();
		const QString& getMicmacDir() const;
		const QString& getStdoutfile() const;
		
		void setEndResult(int endRes);
		void setProgressLabel(const QString& label);
		void setReaderror(const QString& error);
		void setDone(bool b);
	
	private :
		friend void copie(AppliThread* appliThread1, const AppliThread& appliThread2);

		ParamMain* paramMain;
		int 	   endResult;
		bool* 	   annulation;
		QString    progressLabel;
		QString    readerror;
		QString    micmacDir;
		QString    stdoutfile;
		bool 	   done;
};

class InterfPastis;
class PastisThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		enum Avancement { Enregistrement=0, Conversion, Ecriture, FiltrCpls, PtsInteret, ImgsOrientables, Termine };

		PastisThread(ParamMain* pMain, const QString& stdoutfilename, bool* annul, int cpu);
		PastisThread(const PastisThread& pastisThread);
		~PastisThread();

		PastisThread& operator=(const PastisThread& pastisThread);
		void run();
		bool killProcess();
		
	private :
		friend void copie(PastisThread* pastisThread1, const PastisThread& pastisThread2);
		QString defImgOrientables();
		bool corrigeMakefile();
		int cpu_count;
};

class InterfApero;
class Points3DThread;
class Pose;
class AperoThread : public AppliThread
{
   Q_OBJECT
//enregistrement des paramètres saisis dans InterfApero et exécution des calculs d'orientation
	public:
		enum Avancement { Enregistrement=0, Filtrage, AutoCalibration, Poses, PosesLgFocales, DissocCalib, ParamVue3D, Termine };

		AperoThread(ParamMain* pMain, const QString& stdoutfilename, bool* annul, int cpu);
		AperoThread(const AperoThread& aperoThread);
		~AperoThread();

		AperoThread& operator=(const AperoThread& aperoThread);

		void run();
		bool killProcess();
		
	private slots :
		bool points3DThreadFinished(const Points3DThread& points3DThread);

	private :
		friend void copie(AperoThread* aperoThread1, const AperoThread& aperoThread2);
		void trieImgAutoCalib(QList<std::pair<QStringList,QString> >& imagesTriees, QVector<int>& numFocales);
		QString orientationAbsolue();
		QString checkAperoOutStream();

		double echelle;
		QVector<GLdouble> zoneChantier;
		QVector<GLdouble> zoneChantierEtCam;
		int cpu_count;
};
class ParamApero;
class Points3DThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
                Points3DThread();
                Points3DThread(ParamMain* pMain, const QVector<Pose>& cam, bool* annul, const ParamApero& pApero, const vector<string>* fichiers, const vector<pair<string,string> >* pairCam, const QVector<bool>* fishEye);
		Points3DThread(const Points3DThread& points3DThread);
		~Points3DThread();

		Points3DThread& operator=(const Points3DThread& points3DThread);

		void run();
		void setIdx(int i);

		double getEchelle() const;
		const QVector<GLdouble>& getZoneChantier() const;
		const QVector<GLdouble>& getZoneChantierEtCam() const;
		const QVector<Pt3dr>& getEmprise() const;
		
	private :
		friend void copie(Points3DThread* points3DThread1, const Points3DThread& points3DThread2);
		
		QVector<Pose> 	 					 cameras;
		int 			 					 idx;
		double								 echelle;
		QVector<GLdouble>					 zoneChantier;
		QVector<GLdouble>					 zoneChantierEtCam;
		QVector<Pt3dr>						 emprise;
		const ParamApero					*paramApero;
		const vector<string>				*aVN;
		const vector<pair<string,string> >	*aPair;
		const QVector<bool>					*typesCam;	//true si fish-eye
};

class InterfMicmac;
class CarteDeProfondeur;
class MicmacThread : public AppliThread
{
   Q_OBJECT
//enregistrement des paramètres saisis dans InterfApero et exécution des calculs d'orientation
	public:
		enum Avancement { Enregistrement=0, Calcul, Ortho, Termine };

		MicmacThread(ParamMain* pMain, const QString& stdoutfilename, bool* annul);
		void run();
		bool killProcess();
	
	private :
//si type=0, on calcule pour toutes les cartes de pMain->getMasques(), si type=1, on calcule uniquement masque avec les paramètres et noms pour le TA
};

class InterfCartes8B;
class ParamConvert8B;
class Cartes8BThread : public AppliThread
{
   Q_OBJECT
//enregistrement des paramètres saisis dans InterfApero et exécution des calculs d'orientation
	public:
		Cartes8BThread(ParamMain* pMain, const QString& stdoutfilename, ParamConvert8B* pConvert8B, QLabel *label, bool* annul);
		~Cartes8BThread();

		void run();
		bool killProcess();
	
	private :
		ParamConvert8B* paramConvert8B;
		QLabel *infoLabel;
};

class LectureCamThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		LectureCamThread(int num, ParamMain* pMain, Pose* cam);
		~LectureCamThread();
		void run();
		
	private :
                int i;
		Pose* camera;
};

class GLParams;
class Model3DThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		Model3DThread();
		Model3DThread(ParamMain* pMain, GLParams*& pGL);
		Model3DThread(const Model3DThread& model3DThread);
		~Model3DThread();

		void run();
		void setI(int num);
		Model3DThread& operator=(const Model3DThread& model3DThread);
		
	private :
		friend void copie(Model3DThread* model3DThread1, const Model3DThread& model3DThread2);

                int i;
		GLParams* glParam;
};
class Model3DSousThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		Model3DSousThread();
		Model3DSousThread(int num, const QString& carte, ParamMain* pMain, GLParams*& pGL);
		Model3DSousThread(const Model3DSousThread& model3DSousThread);
		~Model3DSousThread();

		void run();
		void setJ(int num);
		Model3DSousThread& operator=(const Model3DSousThread& model3DSousThread);
		
	private :
		friend void copie(Model3DSousThread* model3DSousThread1, const Model3DSousThread& model3DSousThread2);

                int i;
		int j;
		QString numCarte;
		GLParams* glParam;
};

class OrthoThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		OrthoThread();
		OrthoThread(ParamMain* pMain, QLabel *label, bool egal);
		OrthoThread(const OrthoThread& orthoThread);
		~OrthoThread();

		void run();
		void setNumImgRef(QString num);
		OrthoThread& operator=(const OrthoThread& orthoThread);
		
	private :
		friend void copie(OrthoThread* orthoThread1, const OrthoThread& orthoThread2);

		QLabel *infoLabel;
		QString numRef;
		bool egaliser;
};

class ParamNuages;
class ParamPly;
class Convert2PlyThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		Convert2PlyThread();
		Convert2PlyThread(ParamMain* pMain, QLabel *label, const ParamPly& param);
		Convert2PlyThread(const Convert2PlyThread& convert2PlyThread);
		~Convert2PlyThread();

		void run();
		void setParamNuage(const ParamNuages* param);
		Convert2PlyThread& operator=(const Convert2PlyThread& convert2PlyThread);
		
	private :
		friend void copie(Convert2PlyThread* convert2PlyThread1, const Convert2PlyThread& convert2PlyThread2);

		const ParamNuages* paramNuage;
		QLabel *infoLabel;
		const ParamPly* paramPly;
};

class GLWidget;
class NuageThread : public AppliThread
{
   Q_OBJECT

//enregistrement des paramètres saisis dans InterfPastis et exécution des calculs de recherche des points d'intérêt
	public:
		NuageThread(GLWidget* glWidg, int num, int dezoom);
		~NuageThread();

		void run();
		GLuint& getObjectNuag();
		
	private :
                int i;
		int j;
		GLWidget* glWidget;
		GLuint nuageobj;
};

#endif
