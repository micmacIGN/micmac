#include "StdAfx.h"
#include "Affichage.h"


using namespace std;


Affichage::Affichage(ListImg* l, string dossier, int i) : lstImg(l), img0(i), dossierImg(dossier) {}


bool Affichage::Can_open(const char * name)
//renvoie true si le fichier de la photo est trouvé
{
    ELISE_fp fp;
    if (! fp.ropen(name,true))
       return false;

   fp.close();
   return true;
}

Tiff_Im Affichage::GetImage(int img, bool& b, INT& max_val, INT& mult, Pt2di& size) {
	string chemin=(*lstImg).at(img-1-img0).GetChemin();
	string nomImg=(*lstImg).at(img-1-img0).GetNomImg();

	char buf[200];
	sprintf(buf,"%s%s/%s",chemin.c_str(),dossierImg.c_str(),nomImg.c_str());
	cout << "affichage " << buf << "\n";
	ELISE_ASSERT(Can_open(buf),("Pb lecture image "+string(buf)).c_str());
	Tiff_Im Image(buf);

	max_val = 1 << Image.bitpp();
	mult = 255 / (max_val-1);
	size=ElGenFileIm(Image).Sz2();
	b=true;
	return Image;
}

void Affichage::MakeTiff(Bitm_Win aW, string result) {
	char *c = new char[strlen(result.c_str())];
	strcpy(c,result.c_str());
	aW.make_tif(c);
}

void Affichage::AffichePointsImage(ListPt* LstInit, ListPt* LstFin, int img, string result) {
//affiche tous les points de l'image en fonction de leur multiplicité
	if (img-img0>signed((*lstImg).size())) return;
	INT max_val;
	INT mult;
	Pt2di size;
	bool b;
	Tiff_Im Image=GetImage(img, b, max_val, mult, size);
	if (!b) return;

	Elise_Set_Of_Palette aSPal = GlobPal();
	Bitm_Win aW("toto",aSPal,size);
	ELISE_COPY(aW.all_pts(),mult * Image.in(max_val/2),aW.ogray());

	for(ListPt::const_iterator itp=(*LstInit).begin(); itp!=(*LstInit).end(); itp++){
	   	if (Point(*itp).GetCoord().GetImg()==img) {
	   		if(Point(*itp).GetNbHom()==4) aW.draw_circle_abs((*itp).GetPt2dr(), 2.0,Line_St(aW.prgb()(255,0,0),2));
	   		if(Point(*itp).GetNbHom()==3) aW.draw_circle_abs((*itp).GetPt2dr(), 2.0,aW.prgb()(0,255,0));
	   		if(Point(*itp).GetNbHom()==2) aW.draw_circle_abs((*itp).GetPt2dr(), 2.0,aW.prgb()(0,0,255));
	   		if(Point(*itp).GetNbHom()==1) aW.draw_circle_abs((*itp).GetPt2dr(), 2.0,aW.prgb()(130,130,130));
		}
	}
 	for(ListPt::const_iterator itp=(*LstFin).begin(); itp!=(*LstFin).end(); itp++){
	   	if (Point(*itp).GetCoord().GetImg()==img) {
	   		if(Point(*itp).GetNbHom()==4) aW.draw_circle_abs((*itp).GetPt2dr(), 20.0,Line_St(aW.prgb()(255,0,0),2));
	   		if(Point(*itp).GetNbHom()==3) aW.draw_circle_abs((*itp).GetPt2dr(), 20.0,aW.prgb()(0,255,0));
	   		if(Point(*itp).GetNbHom()==2) aW.draw_circle_abs((*itp).GetPt2dr(), 20.0,aW.prgb()(0,0,255));
	   		if(Point(*itp).GetNbHom()==1) aW.draw_circle_abs((*itp).GetPt2dr(), 20.0,aW.prgb()(130,130,130));
		}
	}

	MakeTiff(aW, result);
}

template void Affichage::AffichePointsPaire(list<Point1>::const_iterator itBegin, list<Point1>::const_iterator itEnd, int img, int img2, string result1, string result2, bool segment);
template void Affichage::AffichePointsPaire(ListPt::const_iterator itBegin, ListPt::const_iterator itEnd, int img, int img2, string result1, string result2, bool segment);


