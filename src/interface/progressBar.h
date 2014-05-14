/*
progressBar.h et progressBar.cpp regroupent des fonctions permettant de calculer la valeur de la barre de progression (QProgressDialog) lors des principaux calculs.
L'estimation de la valeur se fait essentiellement par lecture des sorties des calculs ou par recherche des fichiers créés.
La fonction de calcul à appliquer dépend de l'étape du traitement (ParamMain::currentMode, ParamMain::avancement et ParamMain::etape)
*/

#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

#include "all.h"


class Progression : public QThread
{
   Q_OBJECT

//calcul du pourcentage de travail effectué par un thread de calcul et affichage dans la progressDialog
	public:
		Progression();
		~Progression();

		void updatePercent(const ParamMain& paramMain, QProgressDialog* progressdialog, const QString& stdoutfile) const;
		int maxComplete(const ParamMain& paramMain) const;
		int maxComplete(const ParamConvert8B& paramConvert8B) const;
	
	private :
		int dispatch(const ParamMain& paramMain, int pDialogVal, const QString& stdoutfile) const;
		int lastComplete(const ParamMain& paramMain) const;

		int whichDcraw(const ParamMain& paramMain, int pDialogVal) const;
		int whichPastis(const ParamMain& paramMain) const;
		int whichApero(const QString& stdoutfile) const;
		int whichHomol3D(const ParamMain& paramMain) const;
		int whichMicmac(const QString& stdoutfile) const;
		bool doOrthoCurrentCarte(const ParamMain& paramMain) const;
		int whichGrShade(const QString& stdoutfile) const;
		//int whichNuage2Ply(ParamMain* paramMain) const;
};

#endif
