#include "StdAfx.h"
#include "Filtres.h"

using namespace std;


Filtre::Filtre() {}
Filtre::Filtre(list<Point1>* l, BenchQdt<TypeFPRIM<Point>,Point >* b, BenchQdt<TypeFPRIMHomol<Point>,Point >* b2, BenchQdt<TypeFPRIM<PtrPt>,PtrPt >* b3, BenchQdt<TypeFPRIMHomol<PtrPt>,PtrPt >* b4, bool rapide) : tempLstPt(l), bench(b), bench2(b2), bench3(b3), bench4(b4) {
	(*bench).clear();
	(*bench2).clear();
	(*bench3).clear();
	(*bench4).clear();
	for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++) {
	//	if(!rapide) {
			(*bench).insert(*itP,false);
			(*bench2).insert(*itP,false);
	//	} else {
				PtrPt ptr(&*itP);
				(*bench3).insert(ptr,false);
				(*bench4).insert(ptr,false);
	//	}
	}
}

void Filtre::Result(ListPt* lstPtInit){
	for(list<Point1>::const_iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
		if(!((*itP).GetSuppr()))
			(*lstPtInit).ReplacePoint(*itP);
	}
	(*lstPtInit).sort();
}

//---------------------------------------------------------------------------------------------------------------------//

void Filtre::LimitesRecouvrement (REAL d){
	ElSTDNS set<Point> S0;
	REAL recouvrement[4];//xmin, ymin, xmax, ymax
		recouvrement[0]=numeric_limits<int>::max();
		recouvrement[1]=numeric_limits<int>::max();
		recouvrement[2]=0;
		recouvrement[3]=0;
		
	for(list<Point1>::const_iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
		S0.clear();
		Pt2dr pt=(*itP).GetPt2dr();
		(*bench).voisins(pt,d,S0);
		if (S0.size()==1) {continue;}//point  isolé

		if (pt.x<recouvrement[0]) recouvrement[0]=pt.x;
		else if (pt.x>recouvrement[2]) recouvrement[2]=pt.x;
		if (pt.y<recouvrement[1]) recouvrement[1]=pt.y;
		else if (pt.y>recouvrement[3]) recouvrement[3]=pt.y;
	}

	REAL* rec=new REAL[4];
	rec=recouvrement;
	int n=0;
	for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++) {
		if ((*itP).GetSuppr()) {continue;}//point déjà supprimé
		if(!(Point(*itP).GetCoord().isInZone(rec))) {
			(*itP).SetSuppr(true); 
			(*bench).remove(*itP);
			(*bench2).remove(*itP);
			n++;
			cout << "pt faux rec : " << Point(*itP).GetCoord().GetX() << "\n";
		}
	}
	cout << "nb pts faux rec : " << n << "\n";
}

//---------------------------------------------------------------------------------------------------------------------//


template <class ClPt, class TIterator> ClPt* Filtre::FindPoint(TIterator itBegin, TIterator itEnd, const Point& P) {
   for (TIterator itP=itBegin; itP!=itEnd; itP++){
	 // if (Point(P).GetCoord()==Point(*itP).GetCoord() && *(P.begin())==*((*itP).begin())) {
	  if (P==(*itP)) {
		return &(*itP);
	  }
  }	
  return 0;
}

template Point* Filtre::FindPoint(list<Point>::iterator itBegin, list<Point>::iterator itEnd, const Point& P);
template Point1* Filtre::FindPoint(list<Point1>::iterator itBegin, list<Point1>::iterator itEnd, const Point& P);


void Filtre::CoherenceVoisins(list<Point1>::iterator itP, float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax){
	list<Point> S1, S2;

	//recherche des plus proches voisins sur l'image 1
	Pt2dr pt1=(*itP).GetPt2dr();
	(*bench).KPPVois(pt1,S1,aNb1,aDistInit,aFact,aNbMax);

	//recherche des plus proches voisins sur l'image 2
	S2.clear();
	Pt2dr pt2=(*itP).GetPt2drHomol();
	(*bench2).KPPVois(pt2,S2,aNb2,aDistInit,aFact,aNbMax);

	//recherche des voisins cohérents
	int nbCoh=-1;
	for (list<Point>::const_iterator it=S1.begin(); it!=S1.end(); it++) {
		if(FindPoint<Point,list<Point>::iterator>(S2.begin(), S2.end(), (*it))!=0) nbCoh++;
	}

	(*itP).SetCoherence(float(nbCoh)/(float(S1.size()-1)));//S1 et S2 contiennent *itP
}

void Filtre::DistanceAuVoisinage (float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax, bool rapide){		
	if (rapide)
		DistanceAuVoisinageRapide (seuilCoherence, aNb1, aNb2, aDistInit, aFact, aNbMax);
	else {	
		for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
			if ((*itP).GetSuppr()) {continue;}//point déjà supprimé
			CoherenceVoisins(itP, seuilCoherence, aNb1, aNb2, aDistInit, aFact, aNbMax);
		}
		(*tempLstPt).sort();	
	
		int n=0;
		for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
			if ((*itP).GetSuppr()) {continue;}//point déjà supprimé
			CoherenceVoisins(itP, seuilCoherence, aNb1, aNb2, aDistInit, aFact, aNbMax);
			if ((*itP).GetCoherence()<seuilCoherence) { 
				(*itP).SetSuppr(true);
				(*bench).remove(*itP);
				(*bench2).remove(*itP);
				n++;
			}
		}
		cout << "nb pts faux vois : " << n << "\n";
	}
}

//---------------------------------------------------------------------------------------------------------------------//

void Filtre::DistanceAuVoisinageRapide (float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax){	
	time_t timer1=time(NULL);
	for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
		if ((*itP).GetSuppr()) {continue;}//point déjà supprimé	

		list<PtrPt> S1, S2;

		//recherche des plus proches voisins sur l'image 1
		Pt2dr pt1=(*itP).GetPt2dr();
		(*bench3).KPPVois(pt1,S1,aNb1,aDistInit,aFact,aNbMax);
		for (list<PtrPt>::const_iterator it1=S1.begin(); it1!=S1.end(); it1++) {
			(*itP).AddVoisin((*it1).pointeur,1);
		}

		//recherche des plus proches voisins sur l'image 2
		Pt2dr pt2=(*(*itP).begin()).GetPt2dr();
		(*bench4).KPPVois(pt2,S2,aNb2,aDistInit,aFact,aNbMax);	
		for (list<PtrPt>::const_iterator it2=S2.begin(); it2!=S2.end(); it2++) {
			(*itP).AddVoisin((*it2).pointeur,2);
		}
		
		(*itP).CalculeCoherence(true);	
	}
	time_t timer2=time(NULL);
	cout << timer2-timer1 << " secondes" << "\n";
	(*tempLstPt).sort();
	
	int n=0;
	for(list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++){
		if ((*itP).GetSuppr()) {continue;}//point déjà supprimé

		if (itP!=(*tempLstPt).begin()) {
			(*itP).UpDateVoisins();
			(*itP).CalculeCoherence(false);
		}
		if ((*itP).GetCoherence()<seuilCoherence) { 
			(*itP).SetSuppr(true);
			(*bench).remove(*itP);
			(*bench2).remove(*itP);
			n++;
		}
		(*itP).ClearVoisins();
	}
	cout << "nb pts faux vois : " << n << "\n";
}

//---------------------------------------------------------------------------------------------------------------------//


DetSign Filtre::Det(Pt2dr pt1,Pt2dr pt2,Pt2dr pt3) {
	//renvoie le signe du sinus entre pt2 et pt3 (origine pt1) ->(pt1pt2)^(pt1pt2)
	REAL D=(pt2.x-pt1.x)*(pt3.y-pt1.y)-(pt2.y-pt1.y)*(pt3.x-pt1.x);
	if (abs(D)<pow(double(10),-19)) return nul;
	return (D>0) ? pos : neg;
}
bool Filtre::CompDet(Point1* pt1, Point1* pt2, Point1* pt3) {
	//renvoie true si l'ordre des points est conservé
	DetSign D1=Det((*pt1).GetPt2dr(), (*pt2).GetPt2dr(), (*pt3).GetPt2dr());
	DetSign D2=Det((*(*pt1).begin()).GetPt2dr(), (*(*pt2).begin()).GetPt2dr(), (*(*pt3).begin()).GetPt2dr());
	return (D1==D2);
}
void Filtre::InscrCoh(ptrpt* tabpt, bool coherents) {
	//incrémente le nombre d'essais
	ptrpt point=*tabpt;
	for (int i=0; i <4; i++) {
		(*point).IncrCarre(coherents);
		point=*(tabpt++);
	}
}

void Filtre::CoherenceDesCarres (float seuilCoherence, int aNb, double aDistInit, double aFact, int aNbMax, int nbEssais){
//vérifie que les quadrilatères conservent leur convex/concav-ité
	list<Point> S0;
	vector<Point> V0;

	for (list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++) {
		if ((*itP).GetSuppr()) {continue;}//point déjà supprimé
		//recherche des voisins
		S0.clear();
		Pt2dr pt=(*itP).GetPt2dr();
		(*bench).KPPVois(pt,S0,aNb,aDistInit,aFact,aNbMax);

		V0.clear();
		for(list<Point>::iterator it=S0.begin(); it!=S0.end(); it++) 
			V0.push_back(*it);
		
		for (int i=0; i<nbEssais; i++ ) {
			//sélection aléatoire des trois autres points parmi les voisins
			ptrpt pts[4];
			for (int j=0; j<4; j++) pts[j]=&(*itP);
			for (int j=1; j<4; j++) {
				while ( (*(pts[j])==*(pts[max(0,j-1)])) || (*(pts[j])==*(pts[max(0,j-2)])) || (*(pts[j])==*(pts[max(0,j-3)])) ) {
					int k=int(float(rand())/float(RAND_MAX)*float(V0.size()));
					pts[j]=FindPoint<Point1,list<Point1>::iterator>((*tempLstPt).begin(), (*tempLstPt).end(), V0.at(k));
				}
			}
			
			//vérification de la conservation de la convexité du quadrilatère
			ptrpt* tabpt=new ptrpt[4];
			tabpt=pts;
			bool coh=CompDet(pts[0],pts[1],pts[2]);
			if(!coh) {
				InscrCoh(tabpt,coh);
				continue;
			}
			coh=CompDet(pts[0],pts[2],pts[3]);
			if(!coh) {
				InscrCoh(tabpt,coh);
				continue;
			}
			coh=CompDet(pts[1],pts[2],pts[0]);
			if(!coh) {
				InscrCoh(tabpt,coh);
				continue;
			}
			coh=CompDet(pts[1],pts[3],pts[2]);
			if(!coh) {
				InscrCoh(tabpt,coh);
				continue;
			}
			InscrCoh(tabpt,true);
		}
	}
	(*bench).clear();

	int n=0;
	for (list<Point1>::iterator  itP=(*tempLstPt).begin(); itP!=(*tempLstPt).end(); itP++) {
		if ((*itP).GetSuppr()) {continue;}//point déjà supprimé
		if ((*itP).GetCohCarre()<seuilCoherence) {
			(*itP).SetSuppr(true);
			(*bench).remove(*itP);
			(*bench2).remove(*itP);
			n++;
			//cout << "pt faux carre : " << Point(*itP).GetCoord().GetX() << "\n";
		}
	}
	cout << "nb pts faux carre : " << n << "\n";		
}

