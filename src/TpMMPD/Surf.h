#ifndef __SURF_H__
#define __SURF_H__

#include <complex>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "BufferImage.h"

class SurfPoint
{
public:
    SurfPoint(double x=0.,double y=0., double echelle=1., unsigned char sig_lap=0, double hessian=0.):
    _x(x),_y(y),_echelle(echelle),_sig_lap(sig_lap),_hessian(hessian),_orientation(0.)
    {
        descripteur.resize(64);
    }
    
    
    SurfPoint(SurfPoint const &P)
    {
        //std::cout << "Constructeur par recopie"<<std::endl;
        _x=P.x();
        _y=P.y();
        _echelle=P.echelle();
        _sig_lap=P.sig_lap();
        _hessian=P.hessian();
        _orientation=P.orientation();
        descripteur=P.descripteur;
    }
    
    
    inline double x()const
    {return _x;}
    inline double& x()
    {return _x;}
    inline double y()const
    {return _y;}
    inline double& y()
    {return _y;}
    inline double echelle()const
    {return _echelle;}
    inline unsigned char sig_lap()const
    {return _sig_lap;}
    inline double hessian()const
    {return _hessian;}
    
    inline void setOrientation(double orientation)
    {
        _orientation = orientation;
    }
    inline double orientation()const
    {
        return _orientation;
    }
    inline bool operator < (SurfPoint const &p1)const
    {
        return _hessian < p1.hessian();
    }
    
    std::vector<double> descripteur;
    
    double distance(SurfPoint const &P)const;
    
				void applyTransfo(const double shiftX, const double shiftY, const double fact);
    
private:
    double _x;
    double _y;
    double _echelle;
    unsigned char _sig_lap;
    double _hessian;
    double _orientation;
    
};

class SurfHomologue
{
public:
    SurfHomologue(SurfPoint const &p1,SurfPoint const &p2,double distance=0.)
    {
        _x1 = p1.x();
        _y1 = p1.y();
        _x2 = p2.x();
        _y2 = p2.y();
        _distance = distance;
    }
    inline double x1()const
    {return _x1;}
    inline double y1()const
    {return _y1;}
    inline double x2()const
    {return _x2;}
    inline double y2()const
    {return _y2;}
    inline double distance()const
    {return _distance;}
    
				
private:
    double _x1;
    double _y1;
    double _x2;
    double _y2;
    double _distance;
};


class SurfLayer
{
public:
    SurfLayer(int pas, int lobe, int marge, int taille_filtre,int nc, int nl);
    ~SurfLayer();
    void calculerReponsesHessian(BufferImage<double> const &imageIntegrale);
    inline int nl()const
    {
        return _nl;
    }
    inline int nc()const
    {
        return _nc;
    }
    inline int pas()const
    {
        return _pas;
    }
    inline int marge()const
    {
        return _marge;
    }
    inline int taille_filtre()const
    {
        return _taille_filtre;
    }
    inline double reponse(int c,int l)const
    {
        int cs=c/_pas;
        int ls=l/_pas;
        /**/
        if ((cs<0)||(cs>=_nc)||(ls<0)||(ls>=_nl))
        {
            std::cout << "SurfLayer : sortie de tableau "<<cs<<" "<<ls<<" | "<<_nc<<" "<<_nl<<std::endl;
            return 0.;
        }
        /**/
        return _reponse[cs+ls*_nc];
    }
    inline unsigned char sig_lap(int c,int l)const
    {
        int cs=c/_pas;
        int ls=l/_pas;
        /**/
        if ((cs<0)||(cs>=_nc)||(ls<0)||(ls>=_nl))
        {
            std::cout << "SurfLayer : sortie de tableau "<<cs<<" "<<ls<<" | "<<_nc<<" "<<_nl<<std::endl;
            return 0;
        }
        /**/
        return _sig_lap[cs+ls*_nc];
    }
    double* reponse()const
    {
        return _reponse;
    }
    
private:
    
    int _pas;
    // lobe for this filter (filter size / 3)
    int _lobe;
    int _marge;
    int _taille_filtre;
    int _nc;
    int _nl;
    double *  _reponse;
    unsigned char *_sig_lap;
};

/** Classe pour le calcul des points SURF */
class Surf
{
public:
    static const int CST_Table_Filtres [5][4];
    static const int CST_Taille_Filtres [12];
    static const int CST_Pas_SURF[12];
    static const int CST_Lobe_SURF[12];
    static const int CST_Marge_SURF[12];
    static const double CST_Gauss25[7][7];
    
    Surf(BufferImage<unsigned short> const &imageIn,
         int octaves,
         int intervals,
         int init_sample,
         int nbPoints);
    
    
				Surf(BufferImage<unsigned char> const &imageIn,
                     int octaves,
                     int intervals,
                     int init_sample,
                     int nbPoints);
    
    
    std::vector<SurfPoint> vPoints;
    
    bool isExtremum(int c, int l, SurfLayer const &Ech_p,SurfLayer const &Ech_c,SurfLayer const &Ech_n, float seuil);
    bool interpoleExtremum(int c,int l,SurfLayer const & Ech_p,SurfLayer const & Ech_c,SurfLayer const & Ech_n, SurfPoint &Pt);
    bool calculerDescripteur(BufferImage<double> const &imageIntegrale,SurfPoint &Pt);
    
    
    
    
    
    /** Calcul de l'image integrale pour SURF */
    static void computeIntegral(BufferImage<unsigned short> const &imageIn,BufferImage<double> &imageIntegral,int lig_ini=0,int nb_lig=0);
    inline static double integraleBoite(BufferImage<double> const &imageInegrale,int c0,int l0,int nc,int nl, bool verbose=false);
    static void appariement(std::vector<SurfPoint> const &v1, std::vector<SurfPoint> const &v2,std::vector<SurfHomologue> &vApp,float seuil=0.);
    
    
				static void computeIntegral(BufferImage<unsigned char> const &imageIn,BufferImage<double> &imageIntegral,int lig_ini=0,int nb_lig=0);
    
    
private:
    
    static double modelisationAffine(std::vector<SurfHomologue> const &vApp,std::vector<double> &affine, float seuil=0.);
    
    // Parametres repris de CST_SURF (Xing)
    int _num_echelles;
    int _octaves;
    int _intervals;
    int _init_sample;
    float _seuil_extrema;
    int _pas_surf;
    float _orientation;
    bool _is_orientation;
    float _tolerance_affinage;
    int _numptsmodele;
    float _seuil_validite;
    float _Delta_u;
    float _Delta_v;
    int _type_tri;
    
    std::vector<SurfLayer*> _layers;
    
};
#endif
