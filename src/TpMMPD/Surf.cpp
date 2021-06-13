#include <algorithm>
#include <vector>
#include <map>
#include <set>

#include "Surf.h"


const int Surf::CST_Table_Filtres [5][4] = {
    {0,1,2,3}, {1,3,4,5}, {3,5,6,7}, {5,7,8,9}, {7,9,10,11}};
const int Surf::CST_Taille_Filtres [12] = {9,15,21,27,39,51,75,99,147,195,291,387};
const int Surf::CST_Pas_SURF[12] = {1,1,1,1,2,2,4,4,8,8,16,16};
const int Surf::CST_Lobe_SURF[12] = {3,5,7,9,13,17,25,33,49,65,97,129};
const int Surf::CST_Marge_SURF[12] = {5,8,11,14,20,26,38,50,74,98,146,194};
const double Surf::CST_Gauss25[7] [7] = {
    {0.02350693969273,0.01849121369071,0.01239503121241,0.00708015417522,0.00344628101733,0.00142945847484,0.00050524879060},
    {0.02169964028389,0.01706954162243,0.01144205592615,0.00653580605408,0.00318131834134,0.00131955648461,0.00046640341759},
    {0.01706954162243,0.01342737701584,0.00900063997939,0.00514124713667,0.00250251364222,0.00103799989504,0.00036688592278},
    {0.01144205592615,0.00900063997939,0.00603330940534,0.00344628101733,0.00167748505986,0.00069579213743,0.00024593098864},
    {0.00653580605408,0.00514124713667,0.00344628101733,0.00196854695367,0.00095819467066,0.00039744277546,0.00014047800980},
    {0.00318131834134,0.00250251364222,0.00167748505986,0.00095819467066,0.00046640341759,0.00019345616757,0.00006837798818},
    {0.00131955648461,0.00103799989504,0.00069579213743,0.00039744277546,0.00019345616757,0.00008024231247,0.00002836202103}
};

#define CST_SQRT_2PI                     2.506628274631000242

double CST_Gauss(
        double x,
        double y,
        double s)
{
    double s2=s*s;
    return exp(-(x*x+y*y)/(2*s2))/(CST_SQRT_2PI*s);
}

double SurfPoint::distance(SurfPoint const &P)const
{
    //std::cout << "distance : "<<std::endl;
    double d=0.0;
    for(int i=0;i<64;i++)
    {
        //std::cout << "descripteur["<<i<<"] = "<<descripteur[i]<<" | "<<P.descripteur[i]<<std::endl;
        d+=descripteur[i]*P.descripteur[i];
    }
    return 2.*(1.-d);
}

SurfLayer::SurfLayer(int pas, int lobe, int marge, int taille_filtre,int nc, int nl):_pas(pas),_lobe(lobe),_marge(marge),_taille_filtre(taille_filtre),_nc(nc),_nl(nl)
{
    //std::cout << "allocation de "<<_nc*_nl<<" double"<<std::endl;
    _reponse = new double [_nc*_nl];
    if (_reponse==NULL)
        std::cout << "Pb d'allocation dans SurfLayer::SurfLayer "<<std::endl;
    //std::cout << "allocation de "<<_nc*_nl<<" unsigned char"<<std::endl;
    _sig_lap = new unsigned char [_nc*_nl];
    if (_sig_lap==NULL)
        std::cout << "Pb d'allocation dans SurfLayer::SurfLayer"<<std::endl;

}

SurfLayer::~SurfLayer()
{
    if (_reponse)
        delete[] _reponse;
    if (_sig_lap)
        delete[] _sig_lap;
}
void SurfLayer::calculerReponsesHessian(BufferImage<double> const &imageIntegrale)
{
    bool verbose = false;
    double *ptrReponse = _reponse;
    unsigned char *ptrSigLap = _sig_lap;
    int m=(_taille_filtre-1)/2;
    // normalisation factor
    double w = 1./(double)(_taille_filtre*_taille_filtre);
    w *= w;
    if (verbose)
    {
        std::cout << "calculerReponsesHessian : "<<_nl<<" "<<_nc<<std::endl;
        std::cout << "m : "<<m<<" _lobe : "<<_lobe<<" pas : "<<_pas<<std::endl;
    }
    int NCII,NLII;
    NCII = (int)imageIntegrale.numCols();
    NLII = (int)imageIntegrale.numLines();

    int minL = std::max(m+1,_lobe+1);
    int maxL = NLII-std::max(_lobe,-m-1+_taille_filtre);
    int minC = std::max(m+1,_lobe);
    int maxC = NCII-std::max(2*_lobe-1,_taille_filtre);

    minC = minC + (_pas - minC%_pas);
    maxC = maxC - maxC%_pas;
    minL = minL + (_pas - minL%_pas);
    maxL = maxL - maxL%_pas;

    if (verbose) std::cout << "minC : "<<minC<<" maxC : "<<maxC<<std::endl;
    if (verbose) std::cout << "minL : "<<minL<<" maxL : "<<maxL<<std::endl;

    if ((minC>=maxC)||(minL>=maxL))
    {
        //std::cout << "Zone de taille nulle : "<<minC<<" "<<maxC<<" "<<minL<<" "<<maxL<<std::endl;
        for(int l=0;l<_nl;++l)
        {
            //int ll = l*_pas;
            for(int c=0;c<_nc;++c)
            {
                (*ptrReponse) = 0.;
                (*ptrSigLap) = 1;

                ++ptrReponse;
                ++ptrSigLap;
            }
        }
        return;
    }

    for(int l=0;l<_nl;++l)
    {
        int ll = l*_pas;
        if ((ll<minL)||(ll>=maxL))
        {
            for(int c=0;c<_nc;++c)
            {
                (*ptrReponse) = 0.;
                (*ptrSigLap) = 1;

                ++ptrReponse;
                ++ptrSigLap;
            }
        }
        else
        {
            double const * ptrLineDxx1a = imageIntegrale.getLinePtr(ll-_lobe);
            double const * ptrLineDxx1b = imageIntegrale.getLinePtr(ll+_lobe-1);
            double const * ptrLineDxx2a = ptrLineDxx1a;
            double const * ptrLineDxx2b = ptrLineDxx1b;

            double const * ptrLineDyy1a = imageIntegrale.getLinePtr(ll-m-1);
            double const * ptrLineDyy1b = imageIntegrale.getLinePtr(ll-m-1+_taille_filtre);
            double const * ptrLineDyy2a = imageIntegrale.getLinePtr(ll-_lobe/2-1);
            double const * ptrLineDyy2b = imageIntegrale.getLinePtr(ll+_lobe-_lobe/2-1);

            double const * ptrLineDxy1a = imageIntegrale.getLinePtr(ll-_lobe-1);
            double const * ptrLineDxy1b = imageIntegrale.getLinePtr(ll-1);
            double const * ptrLineDxy2a = imageIntegrale.getLinePtr(ll);
            double const * ptrLineDxy2b = imageIntegrale.getLinePtr(ll+_lobe);
            double const * ptrLineDxy3a = ptrLineDxy1a;
            double const * ptrLineDxy3b = ptrLineDxy1b;
            double const * ptrLineDxy4a = ptrLineDxy2a;
            double const * ptrLineDxy4b = ptrLineDxy2b;

            double const * ptrDxx1a1 = ptrLineDxx1a + (-m-1) + minC;
            double const * ptrDxx1a2 = ptrDxx1a1 + _taille_filtre;
            double const * ptrDxx1b1 = ptrLineDxx1b + (-m-1) + minC;
            double const * ptrDxx1b2 = ptrDxx1b1 + _taille_filtre;

            double const * ptrDxx2a1 = ptrLineDxx2a + (-_lobe/2-1) + minC;
            double const * ptrDxx2a2 = ptrDxx2a1 + _lobe;
            double const * ptrDxx2b1 = ptrLineDxx2b + (-_lobe/2-1) + minC;
            double const * ptrDxx2b2 = ptrDxx2b1 + _lobe;

            double const * ptrDyy1a1 = ptrLineDyy1a + (-_lobe) + minC;
            double const * ptrDyy1a2 = ptrDyy1a1 + 2*_lobe-1;
            double const * ptrDyy1b1 = ptrLineDyy1b + (-_lobe) + minC;
            double const * ptrDyy1b2 = ptrDyy1b1 + 2*_lobe-1;

            double const * ptrDyy2a1 = ptrLineDyy2a + (-_lobe) + minC;
            double const * ptrDyy2a2 = ptrDyy2a1 + 2*_lobe-1;
            double const * ptrDyy2b1 = ptrLineDyy2b + (-_lobe) + minC;
            double const * ptrDyy2b2 = ptrDyy2b1 + 2*_lobe-1;


            double const * ptrDxy1a1 = ptrLineDxy1a + minC;
            double const * ptrDxy1a2 = ptrDxy1a1 + _lobe;
            double const * ptrDxy1b1 = ptrLineDxy1b + minC;
            double const * ptrDxy1b2 = ptrDxy1b1 + _lobe;
            double const * ptrDxy2a1 = ptrLineDxy2a - _lobe-1 + minC;
            double const * ptrDxy2a2 = ptrDxy2a1 + _lobe;
            double const * ptrDxy2b1 = ptrLineDxy2b - _lobe-1 + minC;
            double const * ptrDxy2b2 = ptrDxy2b1 + _lobe;
            double const * ptrDxy3a1 = ptrLineDxy3a - _lobe-1 + minC;
            double const * ptrDxy3a2 = ptrDxy3a1 + _lobe;
            double const * ptrDxy3b1 = ptrLineDxy3b - _lobe-1 + minC;
            double const * ptrDxy3b2 = ptrDxy3b1 + _lobe;
            double const * ptrDxy4a1 = ptrLineDxy4a + minC;
            double const * ptrDxy4a2 = ptrDxy4a1 + _lobe;
            double const * ptrDxy4b1 = ptrLineDxy4b + minC;
            double const * ptrDxy4b2 = ptrDxy4b1 + _lobe;

            for(int c=0;c<minC/_pas;++c)
            {
                (*ptrReponse) = 0.;
                (*ptrSigLap) = 1;

                ++ptrReponse;
                ++ptrSigLap;
            }
            for(int c=minC/_pas;c<maxC/_pas;++c)
            {
                // Position dans l'image

                double Dxx =    ((*ptrDxx1a1) + (*ptrDxx1b2) - (*ptrDxx1a2) - (*ptrDxx1b1)) -
                ((*ptrDxx2a1) + (*ptrDxx2b2) - (*ptrDxx2a2) - (*ptrDxx2b1))*3;

                double Dyy =    ((*ptrDyy1a1) + (*ptrDyy1b2) - (*ptrDyy1a2) - (*ptrDyy1b1)) -
                ((*ptrDyy2a1) + (*ptrDyy2b2) - (*ptrDyy2a2) - (*ptrDyy2b1))*3;

                double Dxy =    ((*ptrDxy1a1) + (*ptrDxy1b2) - (*ptrDxy1a2) - (*ptrDxy1b1))+
                ((*ptrDxy2a1) + (*ptrDxy2b2) - (*ptrDxy2a2) - (*ptrDxy2b1))-
                ((*ptrDxy3a1) + (*ptrDxy3b2) - (*ptrDxy3a2) - (*ptrDxy3b1))-
                ((*ptrDxy4a1) + (*ptrDxy4b2) - (*ptrDxy4a2) - (*ptrDxy4b1));

                (*ptrReponse) = (Dxx*Dyy-0.81*Dxy*Dxy)*w;
                (*ptrSigLap) = Dxx*Dyy>0?1:0;

                ++ptrReponse;
                ++ptrSigLap;
                ptrDxx1a1 +=_pas;
                ptrDxx1a2 +=_pas;
                ptrDxx1b1 +=_pas;
                ptrDxx1b2 +=_pas;
                ptrDxx2a1 +=_pas;
                ptrDxx2a2 +=_pas;
                ptrDxx2b1 +=_pas;
                ptrDxx2b2 +=_pas;
                ptrDyy1a1 +=_pas;
                ptrDyy1a2 +=_pas;
                ptrDyy1b1 +=_pas;
                ptrDyy1b2 +=_pas;
                ptrDyy2a1 +=_pas;
                ptrDyy2a2 +=_pas;
                ptrDyy2b1 +=_pas;
                ptrDyy2b2 +=_pas;
                ptrDxy1a1 +=_pas;
                ptrDxy1a2 +=_pas;
                ptrDxy1b1 +=_pas;
                ptrDxy1b2 +=_pas;
                ptrDxy2a1 +=_pas;
                ptrDxy2a2 +=_pas;
                ptrDxy2b1 +=_pas;
                ptrDxy2b2 +=_pas;
                ptrDxy3a1 +=_pas;
                ptrDxy3a2 +=_pas;
                ptrDxy3b1 +=_pas;
                ptrDxy3b2 +=_pas;
                ptrDxy4a1 +=_pas;
                ptrDxy4a2 +=_pas;
                ptrDxy4b1 +=_pas;
                ptrDxy4b2 +=_pas;
            }
            for(int c=maxC/_pas;c<_nc;++c)
            {
                (*ptrReponse) = 0.;
                (*ptrSigLap) = 1;

                ++ptrReponse;
                ++ptrSigLap;
            }
        }
    }
}

double Surf::modelisationAffine(std::vector<SurfHomologue> const &vApp,std::vector<double> &affine, float seuil)
{
    affine.clear();
    affine.resize(6);
    double         M[3][3];
    double         Q[3][3];
    double         B[3][2];
    double             det;

    for(int i=0;i<3;i++) {
        for(int j=0;j<3;j++) {
            M[i][j]=0;
        }
        for(int j=0;j<2;j++) {
            B[i][j]=0;
        }
    }

    for(size_t ip=0;ip<vApp.size();ip++)
    {
        SurfHomologue const &H = vApp[ip];
        if ((seuil==0.)||(H.distance()>seuil)) continue;
        double x1 = H.x1();
        double y1 = H.y1();
        double x2 = H.x2();
        double y2 = H.y2();

        M[0][0]+=x1*x1;
        M[0][1]+=x1*y1;
        M[0][2]+=x1;
        M[1][1]+=y1*y1;
        M[1][2]+=y1;
        M[2][2]+=1.0;

        B[0][0]+=x1*x2;
        B[0][1]+=x1*y2;
        B[1][0]+=y1*x2;
        B[1][1]+=y1*y2;
        B[2][0]+=x2;
        B[2][1]+=y2;
    }
    Q[0][0]=M[1][1]*M[2][2]-M[1][2]*M[1][2];
    Q[1][1]=M[0][0]*M[2][2]-M[0][2]*M[0][2];
    Q[2][2]=M[0][0]*M[1][1]-M[0][1]*M[0][1];
    Q[0][1]=M[0][2]*M[1][2]-M[0][1]*M[2][2];
    Q[0][2]=M[0][1]*M[1][2]-M[0][2]*M[1][1];
    Q[1][2]=M[0][2]*M[0][1]-M[0][0]*M[1][2];
    det=Q[0][0]*M[0][0]+Q[0][1]*M[0][1]+Q[0][2]*M[0][2];
    affine[0]=(Q[0][0]*B[0][0]+Q[0][1]*B[1][0]+Q[0][2]*B[2][0])/det;
    affine[1]=(Q[0][1]*B[0][0]+Q[1][1]*B[1][0]+Q[1][2]*B[2][0])/det;
    affine[2]=(Q[0][2]*B[0][0]+Q[1][2]*B[1][0]+Q[2][2]*B[2][0])/det;
    affine[3]=(Q[0][0]*B[0][1]+Q[0][1]*B[1][1]+Q[0][2]*B[2][1])/det;
    affine[4]=(Q[0][1]*B[0][1]+Q[1][1]*B[1][1]+Q[1][2]*B[2][1])/det;
    affine[5]=(Q[0][2]*B[0][1]+Q[1][2]*B[1][1]+Q[2][2]*B[2][1])/det;

    double somme = 0.;
    int Nb = 0;
    for(size_t ip=0;ip<vApp.size();ip++)
    {
        SurfHomologue const &H = vApp[ip];
        if ((seuil==0.)||(H.distance()>seuil)) continue;
        double x1 = H.x1();
        double y1 = H.y1();
        double x2 = H.x2();
        double y2 = H.y2();
        double rx = affine[0]*x1 + affine[1]*y1 + affine[2] - x2;
        double ry = affine[3]*x1 + affine[4]*y1 + affine[5] - y2;
        somme += rx*rx + ry*ry;
        ++Nb;
    }
    std::cout << "Nombre d'appariements : "<<Nb<<std::endl;
    if (Nb==0)
        return -1;
    return sqrt(somme/(2*Nb));
}


void Surf::appariement(std::vector<SurfPoint> const &v1, std::vector<SurfPoint> const &v2,std::vector<SurfHomologue> &vApp,float seuil)
{
    bool verbose = false;
    if (verbose) std::cout << "appariement"<<std::endl;
    vApp.clear();
    // On calcule toutes les distances
    int N1 = (int)v1.size();
    int N2 = (int)v2.size();
    if ((N1==0)||(N2==0)) return;

    if (verbose) std::cout << "Tableau de "<<N1*N2<<" valeurs"<<std::endl;
    //double *distance = new double[N1*N2];
    double **distance = new double*[N1];
    for(int i=0;i<N1;++i)
        distance[i] = new double[N2];

    // Pour chaque point de v1 on cherche le point le plus proche dans v2
    std::vector<int> v12;
    if (verbose) std::cout << "Recherche directe"<<std::endl;
    for(int i=0;i<N1;++i)
    {
        SurfPoint const &P1 = v1[i];
        double dmin = P1.distance(v2[0]);
        int idmin = 0;
        double *distanceI = distance[i];
        distanceI[0] = dmin;
        for(int j=1;j<N2;++j)
        {
            SurfPoint const &P2=v2[j];
            double d = P1.distance(P2);
            distanceI[j] = d;
            if (d<dmin)
            {
                dmin = d;
                idmin = j;
            }
        }
        //if (verbose) std::cout << "distance min 1->2 : "<<i<<" -> "<<idmin<<std::endl;
        v12.push_back(idmin);
    }
    // Pour chaque point de v2 on cherche le point le plus proche dans v1
    if (verbose) std::cout << "Verification Retour"<<std::endl;
    for(int j=0;j<N2;++j)
    {
        double dmin = distance[0][j];
        int idmin = 0;
        for(int i=1;i<N1;++i)
        {
            double d = distance[i][j];
            if (d<dmin)
            {
                dmin = d;
                idmin = i;
            }
        }
        //if (verbose) std::cout << "distance min 2->1 : "<<j<<" "<<idmin<<std::endl;
        if ((v12[idmin]==j)&&((seuil==0.)||(seuil>dmin)))
        {
            //if (verbose) std::cout << "Appariement de "<<v1[idmin].x()<<" "<<v1[idmin].y()<<" | "<<v2[j].x()<<" "<<v2[j].y()<<" distance : "<<dmin<<std::endl;
            vApp.push_back(SurfHomologue(v1[idmin],v2[j],dmin));
        }
    }
    for(int i=0;i<N1;++i)
        delete[] distance[i];
    delete[] distance;

    //if (seuil==0.)
    return;
    /*
    std::vector<SurfHomologue> vAppF;

    // Estimation d'un modele affine pour le filtrage
    double S = 1.;
    std::vector<double> affine;
    double r = modelisationAffine(vApp,affine,S);
    std::cout << "Seuil : "<<S<<" R : "<<r<<std::endl;
    while(r>seuil)
    {
        S -= S/4.;
        r = modelisationAffine(vApp,affine,S);
        std::cout << "Seuil : "<<S<<" R : "<<r<<std::endl;
        for(size_t ip=0;ip<vApp.size();ip++)
        {
            SurfHomologue const &H = vApp[ip];
            if ((seuil==0.)||(H.distance()>S)) continue;
            double x1 = H.x1();
            double y1 = H.y1();
            double x2 = H.x2();
            double y2 = H.y2();
            double rx = affine[0]*x1 + affine[1]*y1 + affine[2] - x2;
            double ry = affine[3]*x1 + affine[4]*y1 + affine[5] - y2;
            std::cout << "Residus : "<<rx<<" "<<ry<<std::endl;
        }
    }
    for(size_t ip=0;ip<vApp.size();ip++)
    {
        SurfHomologue const &H = vApp[ip];
        if ((seuil==0.)||(H.distance()>S)) continue;
        vAppF.push_back(H);
    }

    vApp.swap(vAppF);

    std::cout << "Fin"<<std::endl;
    */
}

Surf::Surf(BufferImage<unsigned short> const &imageIn,
           int octaves,
           int intervals,
           int init_sample,
           int nbPoints):_octaves(octaves),_intervals(intervals),_seuil_extrema((float).0008),_pas_surf(init_sample)
{
    bool verbose = false;

    if (verbose) std::cout << __FILE__<<" : "<<__LINE__<<" Surf "<<octaves<<" "<<intervals<<" "<<init_sample<<" "<<nbPoints<<std::endl;
    int NC,NL;
    NC = (int)imageIn.numCols();
    NL = (int)imageIn.numLines();

    // nombre de resolutions de calcul des reponses de Hessian
    _num_echelles=CST_Table_Filtres [_octaves-1][_intervals-1]+1;

    // Pour des questions de memoire il faut traiter l'image par bloc de lignes
    // Pour assurer la bonne repartition des points il faut aussi traiter l'image par pave

    // On estime la taille des paves de pixel pour assurer la repartition homogene des points
    int taillePave = 0;
    int nbPointsParPave = 0;
    if (nbPoints)
    {
        nbPointsParPave = std::max(1,nbPoints/100);
        int nbPaves = nbPoints/nbPointsParPave;
        taillePave = (int)(std::sqrt((double)(NC*NL)/(double)nbPaves));
        // Si les paves sont trop petits on prend au min 100 pixels
        if (taillePave<128)
            taillePave = 128;
        nbPaves = (NC*NL)/(taillePave*taillePave);
        nbPointsParPave = nbPoints/nbPaves;
        if (verbose) std::cout << "Il faut en moyenne "<<nbPointsParPave<<" points pour chaque zone de "<<taillePave<<" x "<<taillePave<<" (pixel)"<<std::endl;
    }

    // On estime la taille de bloc de lignes pour le traitement
    int nbMaxLignesUtiles = 2048/*3322*//*1024*/;
    if (verbose) std::cout << "taille de la zone utile des blocs de lignes : "<<nbMaxLignesUtiles<<std::endl;
    int marge = CST_Marge_SURF[_num_echelles-1];
    int nbMaxLignes = nbMaxLignesUtiles + 2*marge;
    if (verbose) std::cout << "taille de la zone avec les marges necessaires au calcul : "<<nbMaxLignes<<std::endl;

    // Il faut maintenant fixer la taille des paves pour qu'ils rentrent bien dans les blocs de lignes
    int taillePaveL = nbMaxLignesUtiles;
    if (taillePave)
    {
        while(taillePaveL>taillePave)
            taillePaveL/=2;
    }
    int taillePaveC = NC;
    if (taillePave)
    {
        taillePaveC = (taillePave*taillePave)/taillePaveL;
    }
    if (verbose) std::cout << "On fait des paves de : "<<taillePaveC<<" x "<<taillePaveL<<" dans des blocs de "<<nbMaxLignesUtiles<<" lignes"<<std::endl;

    for(int lBloc=-marge;lBloc<NL;lBloc+= nbMaxLignesUtiles)
    {
        if (verbose) std::cout << "lBloc : "<<lBloc<<std::endl;

        // Zone de l'image a traiter
        int l0_Image = std::max(0,lBloc);
        //std::cout << "l0_Image : "<<l0_Image<<std::endl;
        int nbl_Image = std::min(nbMaxLignes+lBloc-l0_Image,NL-l0_Image);
        if (verbose) std::cout << "Zone de l'image complete a traiter : "<<l0_Image<<" "<<nbl_Image+l0_Image<<std::endl;

        // Zone utile dans le crop
        int l0_Crop = marge+lBloc-l0_Image;// coord dans le bloc du debut de la zone utile
        int nbl_Crop = std::min(nbMaxLignesUtiles,nbl_Image-l0_Crop);// nombre de lignes utiles a traiter
        if (verbose) std::cout << "Zone utile dans le crop : "<<l0_Crop+l0_Image<<" "<<nbl_Crop+l0_Crop+l0_Image<<std::endl;

        // Preparation de l'image integrale
        BufferImage<double> imageIntegrale;
        //std::cout << "Calcul de l'image integrale"<<std::endl;
        computeIntegral(imageIn,imageIntegrale,l0_Image,nbl_Image);
        //std::cout << "Taille de l'image Integrale : "<<imageIntegrale.numCols()<<" x "<<imageIntegrale.numLines()<<std::endl;
        //std::cout << "fin"<<std::endl;

        //std::cout << "Allocation de l espace memoire pour le calcul des derivees.."<<std::endl;
        // Allocation de l'espace memoire pour le calcul des "derivees"
        if (verbose) std::cout << "_layers.size() avant : "<<_layers.size()<<std::endl;
        for(int i=0;i<_num_echelles;i++)
        {
            //std::cout << "Preparation Hessian pour echelle : "<<i+1<<" / "<<_num_echelles<<std::endl;
            int pas = CST_Pas_SURF[i]*_pas_surf;
            _layers.push_back(new SurfLayer(pas,CST_Lobe_SURF[i],CST_Marge_SURF[i],CST_Taille_Filtres[i],NC/pas,nbl_Image/pas));
            _layers[i]->calculerReponsesHessian(imageIntegrale);
        }
        if (verbose) std::cout << "_layers.size() apres : "<<_layers.size()<<std::endl;
        //std::cout << "Fin de l'allocation"<<std::endl;


        for(int bl=0;(bl*taillePaveL)<nbl_Crop;++bl)
        {
            //std::cout <<"l0_Crop : "<<l0_Crop<<std::endl;
            int nbLignesATraiter = std::min(taillePaveL,nbl_Crop-bl*taillePaveL);
            if (verbose) std::cout << "bl : "<<bl<<" : "<<bl*taillePaveL+l0_Crop+l0_Image<<" | "<<bl*taillePaveL+l0_Crop+l0_Image+nbLignesATraiter<<std::endl;
            for(int bc=0;(bc*taillePaveC)<NC;++bc)
            {
                // Traitement du pave bc,bl
                // Recherche des nbPointsParPave premiers extrema dans la zone utile
                std::multiset<SurfPoint> sPts;

                for(int oct=0;oct<_octaves;++oct)
                {
                    for(int inter=0;inter<2;++inter)
                    {
                        //std::cout << "oct : "<<oct<<" inter : "<<inter<<std::endl;
                        SurfLayer const & Ech_p = *(_layers[CST_Table_Filtres [oct][inter] ]);
                        SurfLayer const & Ech_c = *(_layers[CST_Table_Filtres [oct][inter+1] ]);
                        SurfLayer const & Ech_n = *(_layers[CST_Table_Filtres [oct][inter+2] ]);
                        for(int l=0;l<nbLignesATraiter;l+=Ech_n.pas())
                        {
                            // Position dans le crop
                            int lCrop = l+bl*taillePaveL+l0_Crop;

                            //std::cout << "l : "<<l<<std::endl;
                            for(int c=0;c<taillePaveC;c+=Ech_n.pas())
                            {
                                // Position dans le crop
                                int cCrop = c+bc*taillePaveC;

                                // Position dans l'image complete
                                //int lImage = lCrop+l0_Image;
                                //int cImage = cCrop;

                                if(isExtremum((cCrop)/Ech_n.pas(),(lCrop)/Ech_n.pas(),Ech_p,Ech_c,Ech_n,0.))
                                {
                                    double val = Ech_c.reponse(cCrop,lCrop);
                                    //std::cout << "On a un extremum : "<<cCrop<<" "<<lCrop<<" : "<<val<< " Radiometrie initiale : "<<imageIn(cImage,lImage)<<std::endl;

                                    bool ok = true;
                                    if ((sPts.size()>=(size_t)nbPointsParPave) && ( nbPointsParPave>0))
                                    {
                                        // On teste la valeur
                                        if (val<(*sPts.begin()).hessian())
                                            ok=false;
                                        else
                                        {
                                            sPts.erase(sPts.begin());
                                        }
                                    }
                                    if (ok)
                                    {
                                        //std::cout << "interpoleExtremum..."<<std::endl;
                                        SurfPoint Pt;
                                        if (interpoleExtremum(cCrop/Ech_n.pas(), lCrop/Ech_n.pas(), Ech_p, Ech_c, Ech_n,Pt))
                                        {
                                            calculerDescripteur(imageIntegrale,Pt);
                                            //std::cout << "Point : "<<Pt.x()<<" "<<Pt.y()+l0_Image<<std::endl;
                                            Pt.y()+=l0_Image;
                                            sPts.insert(Pt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                //std::cout << "Nombre de point pour le pave "<<bc<<" x "<<bl<<" : "<<sPts.size()<<std::endl;
                std::multiset<SurfPoint>::const_iterator it,fin=sPts.end();
                for(it=sPts.begin();it!=fin;++it)
                {
                    vPoints.push_back(*it);
                }
            }
        }

        if (verbose) std::cout << "Fin de la recherche de points"<<std::endl;
        for(size_t l=0;l<_layers.size();++l)
        {
            delete _layers[l];
        }
        _layers.clear();
        if (verbose) std::cout << "Fin"<<std::endl;
    }
}

void Surf::computeIntegral(BufferImage<unsigned short> const &imageIn,BufferImage<double> &imageIntegral,int lig_ini,int nb_lig)
{
    int NC,NL/*,NbC*/;
    NC = (int)imageIn.numCols();
    NL = (int)imageIn.numLines();
    //NbC = imageIn.numBands();
    int L0 = lig_ini;
    if ((lig_ini+nb_lig)>NL)
    {
        std::cout << "Surf::computeIntegral ERREUR : "<<lig_ini<<" + "<<nb_lig<<" > "<<NL<<std::endl;
    }
    if (nb_lig>0)
        NL=nb_lig;


    imageIntegral.initialize(NC,NL,1);

    // Traitement de la premiere ligne
    int k=0;
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,0,k)/65536.0;
            imageIntegral(c,0,k) = somme;
        }
    }
    for(int l=1;l<=L0;++l)
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,l,k)/65536.0;
            imageIntegral(c,0,k) = imageIntegral(c,0,k) + somme;
        }
    }

    // Traitement des lignes suivantes
    for(int l=1;l<NL;++l)
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,L0+l,k)/65536.0;
            imageIntegral(c,l,k) = imageIntegral(c,l-1,k) + somme;
        }
    }
}


Surf::Surf(BufferImage<unsigned char> const &imageIn,
           int octaves,
           int intervals,
           int init_sample,
           int nbPoints):_octaves(octaves),_intervals(intervals),_seuil_extrema((float).0008),_pas_surf(init_sample)
{
    bool verbose = false;
    std::cout << "Surf : "<<octaves<<" "<<intervals<<" "<<init_sample<<" "<<nbPoints<<std::endl;

    int NC,NL;
    NC = (int)imageIn.numCols();
    NL = (int)imageIn.numLines();

    // nombre de resolutions de calcul des reponses de Hessian
    _num_echelles=CST_Table_Filtres [_octaves-1][_intervals-1]+1;

    // Pour des questions de memoire il faut traiter l'image par bloc de lignes
    // Pour assurer la bonne repartition des points il faut aussi traiter l'image par pave

    // On estime la taille des paves de pixel pour assurer la repartition homogene des points
    int taillePave = 0;
    int nbPointsParPave = 0;
    if (nbPoints)
    {
        nbPointsParPave = std::max(1,nbPoints/100);
        int nbPaves = nbPoints/nbPointsParPave;
        taillePave = (int)(std::sqrt((double)(NC*NL)/(double)nbPaves));
        // Si les paves sont trop petits on prend au min 100 pixels
        if (taillePave<128)
            taillePave = 128;
        nbPaves = (NC*NL)/(taillePave*taillePave);
        nbPointsParPave = nbPoints/nbPaves;
        if (verbose) std::cout << "Il faut en moyenne "<<nbPointsParPave<<" points pour chaque zone de "<<taillePave<<" x "<<taillePave<<" (pixel)"<<std::endl;
    }

    // On estime la taille de bloc de lignes pour le traitement
    int nbMaxLignesUtiles = 2048/*3322*//*1024*/;
    if (verbose) std::cout << "taille de la zone utile des blocs de lignes : "<<nbMaxLignesUtiles<<std::endl;
    int marge = CST_Marge_SURF[_num_echelles-1];
    int nbMaxLignes = nbMaxLignesUtiles + 2*marge;
    if (verbose) std::cout << "taille de la zone avec les marges necessaires au calcul : "<<nbMaxLignes<<std::endl;

    // Il faut maintenant fixer la taille des paves pour qu'ils rentrent bien dans les blocs de lignes
    int taillePaveL = nbMaxLignesUtiles;
    if (taillePave)
    {
        while(taillePaveL>taillePave)
            taillePaveL/=2;
    }
    int taillePaveC = NC;
    if (taillePave)
    {
        taillePaveC = (taillePave*taillePave)/taillePaveL;
    }
    if (verbose) std::cout << "On fait des paves de : "<<taillePaveC<<" x "<<taillePaveL<<" dans des blocs de "<<nbMaxLignesUtiles<<" lignes"<<std::endl;

    for(int lBloc=-marge;lBloc<NL;lBloc+= nbMaxLignesUtiles)
    {
        //std::cout << "lBloc : "<<lBloc<<std::endl;

        // Zone de l'image a traiter
        int l0_Image = std::max(0,lBloc);
        //std::cout << "l0_Image : "<<l0_Image<<std::endl;
        int nbl_Image = std::min(nbMaxLignes+lBloc-l0_Image,NL-l0_Image);
        if (verbose) std::cout << "Zone de l'image complete a traiter : "<<l0_Image<<" "<<nbl_Image+l0_Image<<std::endl;

        // Zone utile dans le crop
        int l0_Crop = marge+lBloc-l0_Image;// coord dans le bloc du debut de la zone utile
        int nbl_Crop = std::min(nbMaxLignesUtiles,nbl_Image-l0_Crop);// nombre de lignes utiles a traiter
        if (verbose) std::cout << "Zone utile dans le crop : "<<l0_Crop+l0_Image<<" "<<nbl_Crop+l0_Crop+l0_Image<<std::endl;

        // Preparation de l'image integrale
        BufferImage<double> imageIntegrale;
        //std::cout << "Calcul de l'image integrale"<<std::endl;
        computeIntegral(imageIn,imageIntegrale,l0_Image,nbl_Image);
        //std::cout << "Taille de l'image Integrale : "<<imageIntegrale.numCols()<<" x "<<imageIntegrale.numLines()<<std::endl;
        //std::cout << "fin"<<std::endl;

        //std::cout << "Allocation de l espace memoire pour le calcul des derivees.."<<std::endl;
        // Allocation de l'espace memoire pour le calcul des "derivees"
        for(int i=0;i<_num_echelles;i++)
        {
            //std::cout << "Preparation Hessian pour echelle : "<<i+1<<" / "<<_num_echelles<<std::endl;
            int pas = CST_Pas_SURF[i]*_pas_surf;
            _layers.push_back(new SurfLayer(pas,CST_Lobe_SURF[i],CST_Marge_SURF[i],CST_Taille_Filtres[i],NC/pas,nbl_Image/pas));
            _layers[i]->calculerReponsesHessian(imageIntegrale);
        }
        //std::cout << "Fin de l'allocation"<<std::endl;


        for(int bl=0;(bl*taillePaveL)<nbl_Crop;++bl)
        {
            //std::cout <<"l0_Crop : "<<l0_Crop<<std::endl;
            int nbLignesATraiter = std::min(taillePaveL,nbl_Crop-bl*taillePaveL);
            if (verbose) std::cout << "bl : "<<bl<<" : "<<bl*taillePaveL+l0_Crop+l0_Image<<" | "<<bl*taillePaveL+l0_Crop+l0_Image+nbLignesATraiter<<std::endl;
            for(int bc=0;(bc*taillePaveC)<NC;++bc)
            {
                // Traitement du pave bc,bl
                // Recherche des nbPointsParPave premiers extrema dans la zone utile
                std::multiset<SurfPoint> sPts;

                for(int oct=0;oct<_octaves;++oct)
                {
                    for(int inter=0;inter<2;++inter)
                    {
                        //std::cout << "oct : "<<oct<<" inter : "<<inter<<std::endl;
                        SurfLayer const & Ech_p = *(_layers[CST_Table_Filtres [oct][inter] ]);
                        SurfLayer const & Ech_c = *(_layers[CST_Table_Filtres [oct][inter+1] ]);
                        SurfLayer const & Ech_n = *(_layers[CST_Table_Filtres [oct][inter+2] ]);
                        for(int l=0;l<nbLignesATraiter;l+=Ech_n.pas())
                        {
                            // Position dans le crop
                            int lCrop = l+bl*taillePaveL+l0_Crop;

                            //std::cout << "l : "<<l<<std::endl;
                            for(int c=0;c<taillePaveC;c+=Ech_n.pas())
                            {
                                // Position dans le crop
                                int cCrop = c+bc*taillePaveC;

                                // Position dans l'image complete
                                //int lImage = lCrop+l0_Image;
                                //int cImage = cCrop;

                                if(isExtremum((cCrop)/Ech_n.pas(),(lCrop)/Ech_n.pas(),Ech_p,Ech_c,Ech_n,0.))
                                {
                                    double val = Ech_c.reponse(cCrop,lCrop);
                                    //std::cout << "On a un extremum : "<<cCrop<<" "<<lCrop<<" : "<<val<< " Radiometrie initiale : "<<imageIn(cImage,lImage)<<std::endl;

                                    bool ok = true;
                                    if ((sPts.size()>=(size_t)nbPointsParPave) && ( nbPointsParPave>0))
                                    {
                                        // On teste la valeur
                                        if (val<(*sPts.begin()).hessian())
                                            ok=false;
                                        else
                                        {
                                            sPts.erase(sPts.begin());
                                        }
                                    }
                                    if (ok)
                                    {
                                        //std::cout << "interpoleExtremum..."<<std::endl;
                                        SurfPoint Pt;
                                        if (interpoleExtremum(cCrop/Ech_n.pas(), lCrop/Ech_n.pas(), Ech_p, Ech_c, Ech_n,Pt))
                                        {
                                            calculerDescripteur(imageIntegrale,Pt);
                                            //std::cout << "Point : "<<Pt.x()<<" "<<Pt.y()+l0_Image<<std::endl;
                                            Pt.y()+=l0_Image;
                                            sPts.insert(Pt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                //std::cout << "Nombre de point pour le pave "<<bc<<" x "<<bl<<" : "<<sPts.size()<<std::endl;
                std::multiset<SurfPoint>::const_iterator it,fin=sPts.end();
                for(it=sPts.begin();it!=fin;++it)
                {
                    vPoints.push_back(*it);
                }
            }
        }

        if (verbose) std::cout << "Fin de la recherche de points"<<std::endl;
        for(size_t l=0;l<_layers.size();++l)
        {
            delete _layers[l];
        }
        _layers.clear();
        if (verbose) std::cout << "Fin"<<std::endl;
    }
}

void Surf::computeIntegral(BufferImage<unsigned char> const &imageIn,BufferImage<double> &imageIntegral,int lig_ini,int nb_lig)
{
    int NC,NL/*,NbC*/;
    NC = (int)imageIn.numCols();
    NL = (int)imageIn.numLines();
    //NbC = imageIn.numBands();
    int L0 = lig_ini;
    if ((lig_ini+nb_lig)>NL)
    {
        std::cout << "Surf::computeIntegral ERREUR : "<<lig_ini<<" + "<<nb_lig<<" > "<<NL<<std::endl;
    }
    if (nb_lig>0)
        NL=nb_lig;


    imageIntegral.initialize(NC,NL,1);

    // Traitement de la premiere ligne
    int k=0;
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,0,k)/256.0;
            imageIntegral(c,0,k) = somme;
        }
    }
    for(int l=1;l<=L0;++l)
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,l,k)/256.0;
            imageIntegral(c,0,k) = imageIntegral(c,0,k) + somme;
        }
    }

    // Traitement des lignes suivantes
    for(int l=1;l<NL;++l)
    {
        double somme = 0.;
        for(int c=0;c<NC;++c)
        {
            somme += (double)imageIn(c,L0+l,k)/256.0;
            imageIntegral(c,l,k) = imageIntegral(c,l-1,k) + somme;
        }
    }
}

double Surf::integraleBoite(BufferImage<double> const &imageIntegrale,int c0,int l0,int nc,int nl, bool verbose)
{
    int      li=l0-1;
    int      lf=l0+nl-1;
    int      ci=c0-1;
    int      cf=c0+nc-1;

    int NC,NL;
    NC = (int)imageIntegrale.numCols();
    NL = (int)imageIntegrale.numLines();


    if ((ci<0) || (li<0) || (cf>= NC) || (lf>= NL))
    {
        if (verbose) std::cout << "test1 FALSE -> return 0"<<std::endl;
        return 0.;
    }

    const double * lineI = imageIntegrale.getLinePtr(li);
    const double * lineF = imageIntegrale.getLinePtr(lf);
    double A = lineI[ci]+lineF[cf]-lineI[cf]-lineF[ci];
    //double A = imageIntegrale(ci,li)+imageIntegrale(cf,lf)-imageIntegrale(cf,li)-imageIntegrale(ci,lf);
    if (verbose) std::cout << lineI[ci] << " " <<lineF[cf]<<" " <<lineI[cf]<<" " <<lineF[ci]<<std::endl;
    if (A<0)
    {
        if (verbose) std::cout << "test2 FALSE -> return 0"<<std::endl;
        A=0.;
    }
    return A;
}


bool Surf::calculerDescripteur(BufferImage<double> const &imageIntegrale,SurfPoint &Pt)
{
    bool verbose = false;
    if (verbose) std::cout << "Surf::calculerDescripteur du point"<<Pt.x()<<" "<<Pt.y()<<std::endl;
    Pt.setOrientation(0.);
    /*
    if(_is_orientation)
    {
        if (verbose) std::cout << "is_orientation"<<std::endl;
        //CST_SURF_CalculerOrientation(Surf,Pt);
    }
    else
    {
        if (verbose) std::cout << "else"<<std::endl;
        Pt.setOrientation(_orientation);
    }
    */
    if (verbose) std::cout << "orientation : "<<Pt.orientation()<<std::endl;

    double co = cos(Pt.orientation());
    double si = sin(Pt.orientation());

    if (verbose)
        std::cout << "co : "<<co<<" si : "<<si<<std::endl;

    int i=-8;
    int pas_des=(int)floor(Pt.echelle()+0.5);
    size_t iind=0;
    double    cx=-0.5;
    double     cy=0.0;
    double  norme=0.0;
    while(i<12)
    {
        int j=-8;
        i=i-4;
        cx+=1.0;
        cy=-0.5;
        while(j<12)
        {
            double dx,dy,mdx,mdy;
            dx=dy=mdx=mdy=0.0;
            cy+=1.0;
            j=j-4;
            int ix=i+5;
            int jx=j+5;
            int xs = (int)floor(Pt.x() +(-jx*Pt.echelle()*si+ix*Pt.echelle()*co)+0.5);
            int ys = (int)floor(Pt.y() +( jx*Pt.echelle()*co+ix*Pt.echelle()*si)+0.5);
            for(int ik=i;ik<i+9;ik++)
            {
                for(int jk=j;jk<j+9;jk++)
                {
                    int s_x=(int)floor(Pt.x()+(-jk*Pt.echelle()*si+ik*Pt.echelle()*co)+0.5);
                    int s_y=(int)floor(Pt.y()+( jk*Pt.echelle()*co+ik*Pt.echelle()*si)+0.5);
                    double g_s1=CST_Gauss(xs-s_x,ys-s_y,2.5*Pt.echelle());
                    double rx=Surf::integraleBoite(imageIntegrale,s_x,s_y-pas_des,pas_des,2*pas_des)
                              -Surf::integraleBoite(imageIntegrale,s_x-pas_des,s_y-pas_des,pas_des,2*pas_des);
                    double ry=Surf::integraleBoite(imageIntegrale,s_x-pas_des,s_y,2*pas_des,pas_des)
                              -Surf::integraleBoite(imageIntegrale,s_x-pas_des,s_y-pas_des,2*pas_des,pas_des);
                    double rrx=g_s1*(-rx*si + ry*co);
                    double rry=g_s1*(rx*co + ry*si);
                    dx+=rrx;
                    dy+=rry;
                    mdx+=fabs(rrx);
                    mdy+=fabs(rry);
                }
            }
            double g_s2 = CST_Gauss(cx-2.0,cy-2.0,1.5);
            Pt.descripteur[iind++]=dx*g_s2;
            Pt.descripteur[iind++]=dy*g_s2;
            Pt.descripteur[iind++]=mdx*g_s2;
            Pt.descripteur[iind++]=mdy*g_s2;
            /*
            if (iind<Pt.descripteur.size())
                Pt.descripteur[iind++]=dx*g_s2;
            else
                Pt.descripteur.push_back(dx*g_s2);
            if (iind<Pt.descripteur.size())
                Pt.descripteur[iind++]=dy*g_s2;
            else
                Pt.descripteur.push_back(dy*g_s2);
            if (iind<Pt.descripteur.size())
                Pt.descripteur[iind++]=mdx*g_s2;
            else
                Pt.descripteur.push_back(mdx*g_s2);
            if (iind<Pt.descripteur.size())
                Pt.descripteur[iind++]=mdy*g_s2;
            else
                Pt.descripteur.push_back(mdy*g_s2);
                */
            norme+=(dx*dx+dy*dy+mdx*mdx+mdy*mdy)*g_s2*g_s2;
            j+=9;
        }
        i+=9;
    }
    norme =sqrt(norme);
    if (verbose) std::cout << "norme : "<<norme<<std::endl;
    //std::cout << "Pt.descripteur.size() : "<<Pt.descripteur.size()<<std::endl;
    for(iind=0;iind<Pt.descripteur.size();iind++) {
        Pt.descripteur[iind] /= norme;
        //std::cout << Pt.descripteur[iind] << " ";
    }
    //std::cout << std::endl;

    return true;
}

bool Surf::interpoleExtremum(int c,int l,SurfLayer const & Ech_p,SurfLayer const & Ech_c,SurfLayer const & Ech_n, SurfPoint &Pt)
{
    //std::cout << "Surf::interpoleExtremum : "<<c<<" "<<l<<std::endl;
    // Position dans l'image en pleine resolution
    int col = c*Ech_n.pas();
    int lig = l*Ech_n.pas();
    //std::cout << "Coord en pleine resolution : "<<col<<" "<<lig<<std::endl;

    int dpas=Ech_c.pas()-Ech_p.pas();



    double val=Ech_c.reponse(col,lig);
    unsigned char sig_lap=Ech_c.sig_lap(col,lig);
    //std::cout << "val : "<<val<<" sig_lap : "<<sig_lap<<std::endl;

    //std::cout << "preparation des buffers..."<<std::endl;
    double          dxye[3];
    double       mxye[3][3];
    double       qxye[3][3];
    double             X[3];
    //std::cout << "...fin de la preparation des buffers"<<std::endl;


    int pas_n = Ech_n.pas();
    int pas_p = Ech_p.pas();

    dxye[0]=(Ech_c.reponse(col+pas_n,lig)-Ech_c.reponse(col-pas_n,lig))/2.0;
    dxye[1]=(Ech_c.reponse(col,lig+pas_n)-Ech_c.reponse(col,lig-pas_n))/2.0;
    dxye[2]=(Ech_n.reponse(col,lig)-Ech_p.reponse(col,lig))/2.0;
    //std::cout << "dxye : "<<dxye[0]<<" "<<dxye[1]<<" "<<dxye[2]<<std::endl;

    mxye[0][0]=Ech_c.reponse(col+pas_n,lig)+Ech_c.reponse(col-pas_n,lig)-2*val;
    mxye[1][1]=Ech_c.reponse(col,lig+pas_n)+Ech_c.reponse(col,lig-pas_n)-2*val;
    mxye[0][1]=(Ech_c.reponse(col+pas_n,lig+pas_n)
                +Ech_c.reponse(col-pas_n,lig-pas_n)
                -Ech_c.reponse(col-pas_n,lig+pas_n)
                -Ech_c.reponse(col+pas_n,lig-pas_n))/4.0;


    mxye[2][2]=Ech_n.reponse(col,lig)+Ech_p.reponse(col,lig)-2*val;
    mxye[0][2]=(Ech_n.reponse(col+pas_n,lig)
                -Ech_n.reponse(col-pas_n,lig)
                -Ech_p.reponse(col+pas_p,lig)
                +Ech_p.reponse(col-pas_p,lig))/4.0;
    mxye[1][2]=(Ech_n.reponse(col,lig+pas_n)
                -Ech_n.reponse(col,lig-pas_n)
                -Ech_p.reponse(col,lig+pas_p)
                +Ech_p.reponse(col,lig-pas_p))/4.0;
    /*
    //std::cout << "mxye[0] : "<<mxye[0][0]<<" "<<mxye[0][1]<<" "<<mxye[0][2]<<std::endl;
    //std::cout << "mxye[1] : "<<mxye[1][1]<<" "<<mxye[0][2]<<std::endl;
    //std::cout << "mxye[2] : "<<mxye[2][2]<<std::endl;
*/
    qxye[0][0]=mxye[1][1]*mxye[2][2]-mxye[1][2]*mxye[1][2];
    qxye[1][1]=mxye[0][0]*mxye[2][2]-mxye[0][2]*mxye[0][2];
    qxye[2][2]=mxye[0][0]*mxye[1][1]-mxye[0][1]*mxye[0][1];
    qxye[0][1]=mxye[0][2]*mxye[1][2]-mxye[0][1]*mxye[2][2];
    qxye[0][2]=mxye[0][1]*mxye[1][2]-mxye[0][2]*mxye[1][1];
    qxye[1][2]=mxye[0][2]*mxye[0][1]-mxye[0][0]*mxye[1][2];
    /*
    //std::cout << "qxye[0] : "<<qxye[0][0]<<" "<<qxye[0][1]<<" "<<qxye[0][2]<<std::endl;
    //std::cout << "qxye[1] : "<<qxye[1][1]<<" "<<qxye[0][2]<<std::endl;
    //std::cout << "qxye[2] : "<<qxye[2][2]<<std::endl;
*/
    double det=qxye[0][0]*mxye[0][0]+qxye[0][1]*mxye[0][1]+qxye[0][2]*mxye[0][2];

    //std::cout << "det : "<<det<<std::endl;

    if (det==0.)
        return false;

    X[0]=-(qxye[0][0]*dxye[0]+qxye[0][1]*dxye[1]+qxye[0][2]*dxye[2])/det;
    X[1]=-(qxye[0][1]*dxye[0]+qxye[1][1]*dxye[1]+qxye[1][2]*dxye[2])/det;
    X[2]=-(qxye[0][2]*dxye[0]+qxye[1][2]*dxye[1]+qxye[2][2]*dxye[2])/det;

    //std::cout << "X : "<<X[0]<<" "<<X[1]<<" "<<X[2]<<std::endl;

    if(fabs(X[0])>=0.5||fabs(X[1])>=0.5||fabs(X[2])>=0.5) {
        return false;
    }

    //std::cout << "Creation du point : "<<col+X[0]*pas_n<<" "<<lig+X[1]*pas_n<<" : "<<0.1333*(Ech_c.taille_filtre()+X[2]*dpas)<<" | "<<sig_lap<<" | "<<val<<std::endl;
    Pt = SurfPoint(col+X[0]*pas_n,lig+X[1]*pas_n,0.1333*(Ech_c.taille_filtre()+X[2]*dpas),sig_lap,val);
    return true;
}

bool Surf::isExtremum(int c, int l, SurfLayer const &Ech_p,SurfLayer const &Ech_c,SurfLayer const &Ech_n, float seuil)
{
    int m = Ech_n.marge()/Ech_n.pas();
    if ((l<=m)||
        (l>=(Ech_n.nl()-m))||
        (c<=m)||
        (c>=(Ech_n.nc()-m)) )
    {
        return false;
    }

    // Position dans l'image en pleine resolution
    int col = c*Ech_n.pas();
    int lig = l*Ech_n.pas();

    double val = Ech_c.reponse(col,lig);

    if (val<seuil)
        return false;

    for(int dl=-1;dl<=1;++dl)
    {
        for(int dc=-1;dc<=1;++dc)
        {
            if (Ech_n.reponse(col+dc*Ech_n.pas(),lig+dl*Ech_n.pas()) >= val)
            {
                return false;
            }
            if (((dc!=0)||(dl!=0)) && (Ech_c.reponse(col+dc*Ech_n.pas(),lig+dl*Ech_n.pas()) >= val))
            {
                return false;
            }
            if (Ech_p.reponse(col+dc*Ech_n.pas(),lig+dl*Ech_n.pas()) >= val)
            {
                return false;
            }
        }
    }
    return true;

}



void SurfPoint::applyTransfo(const double shiftX, const double shiftY, const double fact)
{
    _x *= fact;
    _y *= fact;
    _x += shiftX;
    _y += shiftY;
}
