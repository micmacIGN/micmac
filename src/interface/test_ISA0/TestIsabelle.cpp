/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

//filtrage des points SIFT trop nombreux

#include "StdAfx.h"
#include "include_isa/PaireImg.h"
#include "include_isa/Image.h"
#include "include_isa/Coord.h"
#include "include_isa/Point.h"
#include "include_isa/Requete.h"
#include "include_isa/Filtres.h"
#include "include_isa/Affichage.h"

using namespace std;


cTplValGesInit<std::string>  aTpl;
int img0=0;
int Image::nbImg=img0;


class cAppliTestIsabelle
{
    public :
	cAppliTestIsabelle(int argc, char** argv, cParamFusionSift* aPSF):
		param(*aPSF)
	{		
		mDir = param.dossier();
		mICNM = cInterfChantierNameManipulateur::StdAlloc
			   (
			       argc,
			       argv,
			       mDir,
			       aTpl
			   );
	}

	void Test0();
    private :

	std::string                       mDir;
	cInterfChantierNameManipulateur * mICNM;
	cParamFusionSift param;

	void Traitement (ListPt* lstPtInit, ListPt* lstPtFin, ListImg* lstImg, int imgpred, BenchQdt<TypeFPRIM<Point2>,Point2 >* bench, Affichage* a);
	void CalculsSurLesPoints (ListPt* lstPtInit, Image* img1, BenchQdt<TypeFPRIM<Point2>,Point2 >* bench);
	void Sortie(ListPt* lstPtInit, Image* img);
};


void cAppliTestIsabelle::Test0()
{
	ListImg lstImg;
	ListPt lstPtInit, lstPtFin;
	list<Point1> lstPtFaux;

	TypeFPRIM<Point2> Pt_of_Point0;
	BenchQdt<TypeFPRIM<Point2>,Point2> bench0(Pt_of_Point0, param.box().Val(), param.NbObjMax().Val(), param.SzMin().Val());
	TypeFPRIM<Point> Pt_of_Point;
	BenchQdt<TypeFPRIM<Point>,Point >  bench(Pt_of_Point, param.box().Val(), param.NbObjMax().Val(), param.SzMin().Val());
	TypeFPRIMHomol<Point> Pt_of_Point_Homol;
	BenchQdt<TypeFPRIMHomol<Point>,Point > bench2(Pt_of_Point_Homol, param.box().Val(), param.NbObjMax().Val(), param.SzMin().Val());
	TypeFPRIM<PtrPt> Pt_of_PtrPt;
	BenchQdt<TypeFPRIM<PtrPt>,PtrPt> bench3(Pt_of_PtrPt, param.box().Val(), param.NbObjMax().Val(), param.SzMin().Val());
	TypeFPRIMHomol<PtrPt> Pt_of_PtrPt_Homol;
	BenchQdt<TypeFPRIMHomol<PtrPt>,PtrPt> bench4(Pt_of_PtrPt_Homol, param.box().Val(), param.NbObjMax().Val(), param.SzMin().Val());

	Affichage a(&lstImg, param.dossierImg(), img0);
	time_t timer1, timer2, timer3, timeri, timerf;

	const std::vector<std::string> * aVN = mICNM->Get("Key-Set-HomolPastisBin");	//vecteur des noms des fichiers
	std::cout << aVN->size() << "\n";
	timeri=time(NULL);
	int nbFich= param.lastfichier().IsInit()? min(signed(aVN->size()),param.lastfichier().Val()) : aVN->size();
	if(nbFich>0) {	
		
		int imgpred=img0;//n° de l'image précédente
			
		//for (int aK=0; aK<nbFich ; aK++) {
		for (int aK=param.firstfichier().Val()-1; aK<nbFich ; aK++) {
			//on ne prend pas les fichiers de points précédemment filtrés
			string s=StdPrefix ((*aVN)[aK]);
			s=StdPostfix(s,'_');
			if(s.compare("filtre")==0) continue;

		timer1=time(NULL);
		
			//nom des images du fichier (*aVN)[aK]
		  	std::pair<std::string,std::string>  aPair = mICNM->Assoc2To1("Key-Assoc-CpleIm2HomolPastisBin", (*aVN)[aK],false);	
			std::cout <<  (*aVN)[aK] << "\n";

			int i1=distance(lstImg.begin(),find(lstImg.begin(),lstImg.end(),aPair.first));
			//traitement des points de l'image précédente
			if (i1!=imgpred ) {
				Traitement (&lstPtInit, &lstPtFin, &lstImg, imgpred, &bench0, &a);
			}

			if (i1==signed(lstImg.size())) {
				Image image1(aPair.first,mDir);	
				image1.SetAdressePt(lstPtFin.size());	
				lstImg.push_back(image1);
			}
			int numImg1=i1+img0+1;
			std::cout <<  "img1 : " << numImg1 << " " << aPair.first << "\n";
			if (i1!=imgpred) lstImg.at(i1).SetAdressePt(lstPtFin.size());

			int i2=distance(lstImg.begin(),find(lstImg.begin(),lstImg.end(),aPair.second));
			if (i2==signed(lstImg.size())) {
				Image image2(aPair.second,mDir);
				image2.SetAdressePt(lstPtFin.size());	
				lstImg.push_back(image2);
			}
			int numImg2=i2+img0+1;
			std::cout <<  "img2 : " << numImg2 << " " << aPair.second << "\n";

			PaireImg p12(numImg2, (*aVN)[aK]);
			PaireImg* itp=lstImg.at(i1).AddPaire(p12);
						
			//lecture du fichier (*aVN)[aK]
			ElPackHomologue aPack = ElPackHomologue::FromFile(mDir+(*aVN)[aK]);
		
			int n=aPack.size();	
			if(n==0) {
				cout << "pas de points SIFT pour cette paire!" << "\n";
				continue;
			}
			(*itp).SetNbPtsInit(n);
	
			//liste des paires de points homologues
			list<Point1> tempLstPt;
			for (ElPackHomologue::const_iterator  itH=aPack.begin(); itH!=aPack.end() ; itH++)
			{				
				Pt2dr pt1(itH->P1());
				Pt2dr pt2(itH->P2());
				Coord coord1((float)itH->P1().x,(float)itH->P1().y,numImg1);
				Coord coord2((float)itH->P2().x,(float)itH->P2().y,numImg2);

				Point1 point(coord1);
				point.AddHom(coord2);
				tempLstPt.push_back(point);
			}
	  		cout << "nb pts part "  << tempLstPt.size() << "\n";

			//filtrage des points faux
			timer1=time(NULL);
			Filtre filtre(&tempLstPt, &bench, &bench2, &bench3, &bench4, param.rapide().Val());
			//if (param.filtre1().Val()) filtre.LimitesRecouvrement (param.distIsol2().Val());
			if (param.filtre2().Val()) filtre.DistanceAuVoisinage ((float)param.seuilCoherenceVois().Val(), param.aNb1().Val(), param.aNb2().Val(), param.aDistInitVois().Val(), param.aFact().Val(), param.aNbMax().Val(),false);
			//if (param.filtre3().Val()) filtre.CoherenceDesCarres (param.seuilCoherenceCarre().Val(), param.aNb().Val(), param.aDistInitVois().Val(), param.aFact().Val(), param.aNbMax().Val(), param.nbEssais().Val());
			timer2=time(NULL);
			cout << timer2-timer1 << " secondes" << "\n";
			filtre.Result(&lstPtInit);
	  		cout << "nb pts part hom "  << lstPtInit.size() << "\n";
			timer3=time(NULL);
			cout << "calc " << timer3-timer2 << " secondes" << "\n";

			for(list<Point1>::const_iterator  itP=tempLstPt.begin(); itP!=tempLstPt.end(); itP++){
				if((*itP).GetSuppr()) 
					lstPtFaux.push_back(*itP);
			}
			
			for(list<caffichPaire>::const_iterator it=param.affichPaire().begin(); it!=param.affichPaire().end(); it++){
				if (numImg1==(*it).image1() && numImg2==(*it).image2()) {
					if((*it).liste()==0) a.AffichePointsPaire(tempLstPt.begin(), tempLstPt.end(), (*it).image1(),(*it).image2(),(*it).fichier1(),(*it).fichier2(),(*it).trait().Val());
					else if((*it).liste()==1) a.AffichePointsPaire(lstPtFaux.begin(), lstPtFaux.end(), (*it).image1(),(*it).image2(),(*it).fichier1(),(*it).fichier2(),(*it).trait().Val());
					else if((*it).liste()==2) a.AffichePointsPaire(lstPtInit.begin(), lstPtInit.end(), (*it).image1(),(*it).image2(),(*it).fichier1(),(*it).fichier2(),(*it).trait().Val());
				}
			}
			lstPtFaux.clear();

			imgpred=i1;
			//traitement des points de la dernière image
			if (aK==nbFich-1) {
				Traitement (&lstPtInit, &lstPtFin, &lstImg, imgpred, &bench0, &a);
			}
		  	cout << "nb pts tot "  << lstPtFin.size() << "\n";

		}//end for fichiers (*aVN)[aK]

	}
		cout << "fini" << "\n";
		timerf=time(NULL);
		cout << timerf-timeri << " secondes" << "\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cAppliTestIsabelle::Traitement (ListPt* lstPtInit, ListPt* lstPtFin, ListImg* lstImg, int imgpred, BenchQdt<TypeFPRIM<Point2>,Point2 >* bench, Affichage* a) {
//traitement et affichage
	(*lstPtFin).clear();
	CalculsSurLesPoints(lstPtInit, &((*lstImg).at(imgpred)), bench);
	for(list<Point2>::const_iterator  itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++){
		if(!((*itP).GetSelect())) continue;//point non pris
		(*lstPtFin).push_back(*itP);
	}
	for(list<caffichImg>::const_iterator  it=param.affichImg().begin(); it!=param.affichImg().end(); it++){
		if (imgpred+1==(*it).image()) {
			(*a).AffichePointsImage(lstPtInit, lstPtFin, (*it).image(), (*it).fichier());
		}
	}
	for(list<caffichPaire>::const_iterator it=param.affichPaire().begin(); it!=param.affichPaire().end(); it++){
		if (imgpred+1==(*it).image1()) {
			if((*it).liste()==3) (*a).AffichePointsPaire((*lstPtFin).begin(), (*lstPtFin).end(), (*it).image1(),(*it).image2(),(*it).fichier1(),(*it).fichier2(),(*it).trait().Val());
		}
	}
	(*lstPtInit).clear();
}

void cAppliTestIsabelle::CalculsSurLesPoints (ListPt* lstPtInit, Image* img1, BenchQdt<TypeFPRIM<Point2>,Point2 >* bench) {
//filtrage des points trop nombreux
	ElSTDNS set<Point2> S0;
	//ajout des points isolés
	(*bench).clear();

	for(list<Point2>::iterator  itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++){
		if ((*img1).GetNum()!=(*itP).GetCoord().GetImg())
			cout << "pb1 " << "\n"; 

		S0.clear();
		Pt2dr pt=(*itP).GetPt2dr();
		(*bench).voisins(pt,param.distIsol().Val(),S0);
		if (S0.size()==0) {//point isolé
			(*itP).SetSelect(true); //le point est pris	
			(*bench).insert(*itP);

			//pour chaque homologue du point, mise à jour des informations sur la paire correspondante
			for(list<Coord>::const_iterator  itC=(*itP).begin(); itC!=(*itP).end(); itC++){
				int numImg2=(*itC).GetImg();
				PaireImg* itp=&(*find((*img1).begin(),(*img1).end(),numImg2));
				(*itp).RecalculeAddPt((*itC).GetX(),(*itC).GetY(),(float)param.mindistalign().Val());
			}
		}
	}//end for set<Point2>::const_iterator  itP

		int m=0;
		for(list<Point2>::iterator  itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++)
			if((*itP).GetSelect()) m++;
	  	cout << "nb pts int1 "  << m << "\n";

//---------------------------------------------------------------------------------------------------------------------//


	//ajout des points manquants des paires d'images
	for(list<Point2>::iterator itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++){
		if((*itP).GetSelect()) continue;//le point est déjà pris
		bool b=false;
		for(list<Coord>::const_iterator  itC=(*itP).begin(); itC!=(*itP).end(); itC++){
			int numImg2=(*itC).GetImg();
			PaireImg* itp=&(*find((*img1).begin(),(*img1).end(),numImg2));
			if ((*itp).GetNbPts()<param.ptppi().Val() || (*itp).GetIsAlign()) {
				b=true;
				break;
			}		
		}
		if (b) {
			for(list<Coord>::const_iterator  itC=(*itP).begin(); itC!=(*itP).end(); itC++){
				int numImg2=(*itC).GetImg();
				PaireImg* itp=&(*find((*img1).begin(),(*img1).end(),numImg2));
				(*itp).RecalculeAddPt((*itC).GetX(),(*itC).GetY(),(float)param.mindistalign().Val());
			}
			(*itP).SetSelect(true); //le point est pris
		}	
	}//end for set<Point2>::const_iterator itP

		m=0;
		for(list<Point2>::iterator  itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++)
			if((*itP).GetSelect()) m++;
	  	cout << "nb pts int2 "  << m << "\n";

	Sortie(lstPtInit,img1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cAppliTestIsabelle::Sortie (ListPt* lstPtInit, Image* img) {
//écriture des points retenus dans un fichier
	ElPackHomologue *aPack = new ElPackHomologue[(*img).GetNbPaires()];
	for(list<Point2>::iterator  itP=(*lstPtInit).begin(); itP!=(*lstPtInit).end(); itP++) {
		if(!(*itP).GetSelect()) continue;
		for(list<Coord>::iterator  itC=(*itP).begin(); itC!=(*itP).end(); itC++) {
			ElCplePtsHomologues aCple ((*itP).GetPt2dr(),(*itC).GetPt2dr());
			int i=distance((*img).begin(),find((*img).begin(),(*img).end(),(*itC).GetImg()));
			aPack[i].Cple_Add(aCple);
		}
	}

	for (int i=0; i<(*img).GetNbPaires(); i++) {
		ElSTDNS string s = StdPrefix((*img).at(i).GetNomFichier());
		string s2 = (*img).GetChemin()+s+"_"+param.extensionSortie().Val()+".dat";
		aPack[i].StdPutInFile(s2);
	}
	delete [] aPack;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void TestW() {
}

extern const char * theNameVar_addon_ParamChantierPhotogram[];

int main(int argc,char ** argv)
{
	
   AddEntryStringifie
   (
        "addon_ParamChantierPhotogram.xml",
         theNameVar_addon_ParamChantierPhotogram,
         true
   );

	std::cout << "ARGC " << argc << "\n";
	for (int aK =0 ; aK< argc ; aK++) //verification de nombre d'arguments "in"
		std::cout << "ARGV[" << aK << "]=" << argv[aK] << "\n";
	ELISE_ASSERT(argc>=2,"Pas assez d'arguments");
	cParamFusionSift aPSF = StdGetObjFromFile<cParamFusionSift>
                              ( string(argv[1]),
                                  "addon_ParamChantierPhotogram.xml",
                                  "ParamFusionSift",
                                  "ParamFusionSift"
                              );
      	MakeFileXML(aPSF,"tata.xml");
	cAppliTestIsabelle aAP(argc,argv,&aPSF);//lancement de l'application
	aAP.Test0();
	return 0;
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
