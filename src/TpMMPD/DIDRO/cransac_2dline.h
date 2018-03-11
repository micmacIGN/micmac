#ifndef CRANSAC_2DLINE_H
#define CRANSAC_2DLINE_H
#include "StdAfx.h"
//pour utiliser MedianSup sur un vecteur
#include "../../../include/ext_stl/numeric.h"


enum ModelForm {
   a_plus_bx,bx
};


class c2DLineModel
{
public:
    // constructeur du modèle ligne 2D à partir de 2 points
    c2DLineModel();
    c2DLineModel(Pt2dr aP1, Pt2dr aP2);
    // je construit l'objet en connaissant a et b
    c2DLineModel(double a, double b);
    // à partir d'un point, a=0
    c2DLineModel(Pt2dr aP1);
    // on donne une liste d'observations et éventuellement un pourcentage d'outliers à baquer et on demande de calculer la distance à la droite
    void computeCout(std::vector<Pt2dr> * aObs,int aPropOutlier=3);
    void computeCout(std::vector<Pt2dr> * aObs,std::vector<double> * aPond,int aPropOutlier=3);
    double getCout(){return mError;}
    double getA(){return mA;}
    double getB(){return mB;}
    double predict(double aX);

private:
    double mA,mB;// coeficient du modèle
    ModelForm mForme;
    double mError; // cout sur le modèle
    int mPropOutliers; // proportion d'outliers
    bool mQuiet;
    double distPoint2Model(Pt2dr aPt);
     // sera utilisé pour calcul du cout
};

class cRansac_2dline
{
public:
    cRansac_2dline();
    // création de l'objet à partir des observations, définition de la forme du modèle, d'un vecteur de pondération
    cRansac_2dline(std::vector<Pt2dr> * aObs,ModelForm model,int aNbInt);
    cRansac_2dline(std::map<int,Pt2dr> * aObsMap,ModelForm model,int aNbInt);
    //cRansac_2dline(std::vector<Pt2dr> aObs,ModelForm model, std::vector<double> aPonderation);
    ~cRansac_2dline();

    void adjustModel(int nbIter=0,int prcOutliers=0);
    // renvoie une prédiction du modèle ajusté précédement
    double predict(double aX) {return mBestModel.predict(aX);}
    // donne des info sur le modele
    void affiche();
    void setDebugMode(bool aBool){mQuiet=!aBool;}
    c2DLineModel getModel(){return mBestModel;}

private:
    c2DLineModel mBestModel;
    //c2DLineModel mCurrentModel;
    ModelForm mModelForm;
    int mNbIt, mPrcOutliers,mNbItConvergence; // nombre d'itération, pourcentage d'outliers qu'on jette lors du calcul du coût.
    std::vector<Pt2dr> * mObs;
    std::vector<double> mPond;
    bool mQuiet;

    c2DLineModel oneIteration(std::vector<Pt2dr> * aObs); // calcule le cout de myCurrentModel sur les observation

};


// ajuster une droite par LSQ

class cLSQ_2dline
{
public:
    cLSQ_2dline();
    // création de l'objet à partir des observations, définition de la forme du modèle
    cLSQ_2dline(std::vector<Pt2dr> * aObs,ModelForm model=a_plus_bx);
    cLSQ_2dline(std::vector<Pt2dr> * aObs,std::vector<double> * aPond,ModelForm model=a_plus_bx);
    cLSQ_2dline(std::map<int,Pt2dr> * aObsMap, std::map<int,double> * aPondMap, ModelForm model=a_plus_bx);
    ~cLSQ_2dline();

    void adjustModelL2();//L2
    void adjustModelL1();//L1
    // renvoie une prédiction du modèle ajusté précédement
    double predict(double aX) {return mModel.predict(aX);}
    // donne des info sur le modele
    void affiche();
    c2DLineModel getModel(){return mModel;}

private:
    c2DLineModel mModel;
    ModelForm mModelForm;
    //std::vector<Pt2dr> * mObs;
    std::vector<std::pair<double *,Pt2dr *>> pObs; // pointers to ponderation and observation
    double mPond;
    bool mOk;
};


#endif // CRANSAC_2DLINE_H
