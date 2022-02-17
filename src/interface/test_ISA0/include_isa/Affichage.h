#ifndef DEF_AFFICHE
#define DEF_AFFICHE

#include "Point.h"
#include "Image.h"

class Affichage 
{
	public :
		Affichage(ListImg* lstImg, string dossier, int img0=0);

		template <class DataIt> void AffichePointsPaire(DataIt itBegin, DataIt itEnd, int img, int img2, string result1, string result2, bool segment);
		void AffichePointsImage(ListPt* LstInit, ListPt* LstFin, int img, string result);

	private :
		ListImg* lstImg;
		int img0;
		string dossierImg;

		bool Can_open(const char * name);
		Tiff_Im GetImage(int img, bool& b, INT& max_val, INT& mult, Pt2di& size);
		void MakeTiff(Bitm_Win aW, string result);
};





template <class DataIt> void Affichage::AffichePointsPaire(DataIt itBegin, DataIt itEnd, int img, int img2, string result1, string result2, bool segment) {
//affiche les points retenus pour la paire d'image sur les photos
	if (max(img,img2)-img0>signed((*lstImg).size())) return;
	INT max_val;
	INT mult;
	Pt2di size1;
	Pt2di size2;
	bool b;
	Elise_Set_Of_Palette aSPal = GlobPal();

	Tiff_Im Image1=GetImage(img, b, max_val, mult,size1);
	Bitm_Win aW("toto",aSPal,size1);
	if (!b) return;
	ELISE_COPY(aW.all_pts(),mult * Image1.in(max_val/2),aW.ogray());

	Tiff_Im Image2=GetImage(img2, b, max_val, mult,size2);
	Bitm_Win bW("toto",aSPal,size2);
	if (!b) return;
	ELISE_COPY(bW.all_pts(),mult * Image2.in(max_val/2),bW.ogray());

	for(DataIt itp=itBegin; itp!=itEnd; itp++){
	   	if (Point(*itp).GetCoord().GetImg()==img)
		 	for(list<Coord>::const_iterator  itc=(*itp).begin(); itc!=(*itp).end(); itc++)
			   	if ((*itc).GetImg()==img2){
	   				aW.draw_circle_abs((*itp).GetPt2dr(), 50.0,Line_St(aW.prgb()(255,0,0),50));
	   				bW.draw_circle_abs((*itc).GetPt2dr(), 20.0,Line_St(aW.prgb()(255,0,0),50));
					if (segment) {
						REAL x1=(*itp).GetPt2dr().x;
						REAL y1=(*itp).GetPt2dr().y;
						REAL x2=(*itc).GetPt2dr().x;
						REAL y2=(*itc).GetPt2dr().y;
						REAL x=((size1.y-y1)*x2+y2*x1)/(y2-y1+size1.y);
		   				aW.draw_seg((*itp).GetPt2dr(),Pt2dr(x,size1.y),Line_St(aW.prgb()(255,0,0),2));
		   				bW.draw_seg(Pt2dr(x,0),(*itc).GetPt2dr(),Line_St(aW.prgb()(255,0,0),2));
					}
					break;
				}
	}

	MakeTiff(aW, result1);
	MakeTiff(bW, result2);
}





#endif
