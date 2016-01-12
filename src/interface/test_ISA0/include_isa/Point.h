#ifndef DEF_POINT
#define DEF_POINT

#include "Coord.h"

class Point : public std::list<Coord>
{
	public:
		Point();
		Point(Coord c);
		Point(const Point& P);

		void SetCoord(Coord c);

		Coord GetCoord() const;
		int GetNbHom() const;

		void AddHom(Coord c);

		Pt2dr GetPt2dr() const;
		 Pt2dr GetPt2drHomol() const;

	private:
		Coord coord;
		int nbHom;
};


class Point1 : public Point
{
	public:
		Point1(Coord c);

		void SetSuppr(bool b);
		void SetCoherence(float coh);
		void IncrCarre(bool b);

		Coord GetCoord() const;
		bool GetSuppr() const;
		float GetCoherence() const;
		float GetCohCarre() const;

		bool operator < (const Point1 & p1) const;
		bool operator == (const Point1 & p1) const;
		Pt2dr GetPt2drHomol() const;

		void AddVoisin(Point1* P, int i);//1:img1, 2:img2
		void UpDateVoisins();
		void ClearVoisins();
		void CalculeCoherence(bool unicite);

	private:
		//filtrage des points faux
		bool suppr;
		float coherence;
		int carre[2];//carre[0] : les bons essais , carre[1] : les mauvais essais
		std::pair<std::list<Point1*>,std::list<Point1*> > voisins;	//pour la cohérence au voisinage accélérée, pointeurs vers tempLstPt

		list<Point1*>* GetListVoisins(int i);
  		Point1** FindPoint(list<Point1*>::iterator itBegin, list<Point1*>::iterator itEnd);
};

class PtrPt
{
	public:
		Point1* pointeur;

		PtrPt(Point1* P);
		PtrPt();
		Pt2dr GetPt2dr() const;
		Pt2dr GetPt2drHomol() const;

		bool operator < (const PtrPt & p1) const;
		bool operator == (const PtrPt & p1) const;
};

class Point2 : public Point
{
	public:
		Point2(Point1 P);

		void SetSelect(bool b);

		bool GetSelect() const;

		bool operator < (const Point2 & p2) const;
		bool operator == (const Point2 & p2) const;
		bool operator == (const Coord & c) const;

	private:
		bool select;
};



class ListPt : public std::list<Point2>
{
	public:
		ListPt();
		void ReplacePoint(Point1 pt);
};

#endif
