#ifndef DEF_FILTRE
#define DEF_FILTRE


#include "Point.h"
#include "Requete.h"

typedef enum {nul=0,pos=1,neg=2}  DetSign;
typedef Point1* ptrpt;

//class BenchQdt<TypeFPRIM<Point>,Point>;
class Filtre 
		//fonctions de filtrage des points faux
{
	public :
		Filtre();
		Filtre(std::list<Point1>* l, BenchQdt<TypeFPRIM<Point>,Point >* b, BenchQdt<TypeFPRIMHomol<Point>,Point >* b2, BenchQdt<TypeFPRIM<PtrPt>,PtrPt >* b3, BenchQdt<TypeFPRIMHomol<PtrPt>,PtrPt >* b4, bool rapide);

		void LimitesRecouvrement (REAL d);
		void DistanceAuVoisinage (float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax, bool rapide);
		//void CoherenceDesCarres (REAL d1, int nbMinVoisins, int nbEssais, float seuilCoherence);
		void CoherenceDesCarres (float seuilCoherence, int aNb, double aDistInit, double aFact, int aNbMax, int nbEssais);

		void Result(ListPt* lstPtInit);

	private :
		std::list<Point1>* tempLstPt;
		BenchQdt<TypeFPRIM<Point>,Point >* bench;
		BenchQdt<TypeFPRIMHomol<Point>,Point >* bench2;
		BenchQdt<TypeFPRIM<PtrPt>,PtrPt >* bench3;
		BenchQdt<TypeFPRIMHomol<PtrPt>,PtrPt >* bench4;

		template <class ClPt, class TIterator> ClPt* FindPoint(TIterator itBegin, TIterator itEnd, const Point& P) ;
		void CoherenceVoisins(std::list<Point1>::iterator itP, float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax);
		void DistanceAuVoisinageRapide (float seuilCoherence, int aNb1, int aNb2, double aDistInit, double aFact, int aNbMax);
		DetSign Det(Pt2dr pt1,Pt2dr pt2,Pt2dr pt3);
		bool CompDet(Point1* pt1, Point1* pt2, Point1* pt3);
		void InscrCoh(ptrpt* tabpt, bool coherents);
};
//bool SortPtByCoh(Point1 P1, Point1 P2);

#endif
