#include "cransac_2dline.h"

cRansac_2dline::cRansac_2dline():mObs(0)
{

}

cRansac_2dline::~cRansac_2dline()
{
   // les observation ne sont pas detruite  car elle sont encore utilisées en dehors de cet objet
   //delete mObs;
}

cRansac_2dline::cRansac_2dline(std::vector<Pt2dr> * aObs, ModelForm model, int aNbInt):
    mModelForm(model),
    mNbIt(aNbInt),
    mPrcOutliers(10),
    mNbItConvergence(0),
    mObs(aObs),
    mQuiet(true)
{
    //std::cout << "initialisation d'un objet cRansac_2dline \n";
    // mettre les pondérations à 1
    for (unsigned int i(0) ;i< mObs->size(); i++)
    {
        mPond.push_back(1);
    }
}


cRansac_2dline::cRansac_2dline(std::map<int,Pt2dr> * aObsMap, ModelForm model, int aNbInt)
{
    std::vector<Pt2dr> aObs;

    for (std::map<int, Pt2dr>::iterator it=aObsMap->begin(); it!=aObsMap->end(); ++it)
        aObs.push_back(it->second);
    cRansac_2dline(&aObs,model,aNbInt);
}


void cRansac_2dline::affiche()
{
    // affiche les résumés du modèle
    std::cout << "Modele ligne 2D ransac - summary \n";
    // test si l'ajustement a déjà eu lieu
    if(mNbItConvergence>0)
    {
    std::cout << "Itérations ;" << mNbIt << ". Observations :" << mObs->size() <<". Meilleur modèle déterminé à la " << mNbItConvergence << " ieme itérations, cout de " << mBestModel.getCout() << ".\n";
    std::cout << "a + b * x = y     a=" << mBestModel.getA() << ", b=" << mBestModel.getB() << ".\n";
    } else std::cout << "l'ajustement n'as pas encore été effectué.\n";

}

void cRansac_2dline::adjustModel(int nbIter,int prcOutliers)
{
    // la valeur par défaut des deux arguments est 0
    if (nbIter!=0) mNbIt=nbIter;
    if (prcOutliers!=0) mPrcOutliers=prcOutliers;

    int it(0);
    // RANSAC
    do {

        // pioche 2 points au hazard parmi les observations
        int aRandom1(0),aRandom2(0);
        aRandom1=rand() % (mObs->size()-1);
        aRandom2=rand() % (mObs->size()-1);

        if (!mQuiet) std::cout << "ransac itération " << it+1 << ",  random points num " << aRandom1 << " , " << mObs->at(aRandom1) << "  et num " << aRandom2 << " , " << mObs->at(aRandom2) << "\n";

        // test si tout ça ne bug pas trop en effectuant une itération
        c2DLineModel currentM(mObs->at(aRandom1),mObs->at(aRandom2));
        currentM.computeCout(mObs);

        if (it==0) mBestModel=currentM;


        //if (!mQuiet) std::cout << "cout actuel : " << mBestModel.getCout() << "\n";

        // si le cout de ce modèle est inférieur à celui du bes model, on le garde
        if(currentM.getCout()<mBestModel.getCout())
        {
            if (!mQuiet) std::cout << "ransac itération " << it+1 << ", minimisation du cout, " << mBestModel.getCout() << "-->" <<currentM.getCout() << "\n";
            mBestModel=currentM;
            mNbItConvergence=it+1;
        }

        it++;
    } while (it<mNbIt);

}




/* -------------------------------------- classe Ligne 2D model --------------------------------------- */


c2DLineModel::c2DLineModel():mA(0),mB(1),mForme(a_plus_bx),mError(10000000),mPropOutliers(3)
{
}

c2DLineModel::c2DLineModel(double a, double b):mA(a),mB(b),mForme(a_plus_bx),mError(10000000)
{
}


c2DLineModel::c2DLineModel(Pt2dr aP1, Pt2dr aP2):mA(0),mB(0),mForme(a_plus_bx),mError(10000000),mPropOutliers(3)
{
        // pente
        if (aP2.x!=aP1.x)
        {
        mB=(aP2.y-aP1.y)/(aP2.x-aP1.x);
        } else mB=0; // 0 au hasard, ransac va baquer le modèle et puis c'est tout
        // intersect
        mA=aP1.y-mB*aP1.x;
}

c2DLineModel::c2DLineModel(Pt2dr aP1):mForme(bx),mError(10000000),mPropOutliers(3)
{
        // intersect
        mA=0;
        // pente
        if (aP1.x!=0)
        {
        mB=aP1.y/aP1.x;
        }
}


double c2DLineModel::predict(double aX)
{
    return (mA+mB*aX);
}

double c2DLineModel::distPoint2Model(Pt2dr aPt)
{
    return (aPt.y-predict(aPt.x));
}

void c2DLineModel::computeCout(std::vector<Pt2dr> * aObs,int aPropOutlier)
{
    std::vector<double> aPond;
    for (unsigned int i(0); i<aObs->size();i++) aPond.push_back(1);

    computeCout(aObs,&aPond, aPropOutlier);
}


void c2DLineModel::computeCout(std::vector<Pt2dr> * aObs,std::vector<double> * aPond,int aPropOutlier)
{

    if (aObs->size()==aPond->size())
    {
    std::vector<double> couts;

    // boucle sur chacunes des informations
    for (unsigned int i(0) ;  i< aObs->size() ; i++)
    {
        Pt2dr uneObs(aObs->at(i));
        couts.push_back(std::abs(distPoint2Model(uneObs))*(aPond->at(i)));
       // std::cout << "Cout calculé pour l'observation" << uneObs << " de " << std::abs(distPoint2Model(uneObs)) <<"\n";
    }

    // enleve les n outliers supposés
    // trie par ordre ascendant
    std::sort (couts.begin(), couts.end());
    int nbOutliers=aPropOutlier*(aObs->size())/100;
    // moyenne des distances en excluant les outliers présumés
    double sum = 0;
    for (vector<double>::iterator it = couts.begin(); it != couts.begin()+(couts.size()-nbOutliers); ++it) {
        sum += *it;
    }
    double cout=sum/(couts.size()-nbOutliers);

    // modifier les attributs
    mError=cout;
    mPropOutliers=aPropOutlier;
    } else std::cout << "RANSAC 2dline : Vecteurs de pondérations de taille différente que vecteur observation.\n";

}





//                   ajustement droite par LSQ                       //

cLSQ_2dline::cLSQ_2dline(std::vector<Pt2dr> * aObs,ModelForm model):
mModelForm(model),
mPond(1),
mOk(0)
{

    for (unsigned int i(0) ; i<aObs->size();i++)
    {
      pObs.push_back(std::make_pair(&mPond,&aObs->at(i)));
    }
}

cLSQ_2dline::cLSQ_2dline(std::vector<Pt2dr> * aObs,std::vector<double> * aPond,ModelForm model):
mModelForm(model),
mOk(0)
{
    // check that observation and ponderation have the same length
    if(aPond->size()!=aObs->size())
    {std::cout  <<"Warning: Ponderation vector have not the same length than observations.\n";}
    if((aPond->size()==0 )|(aObs->size()==0))
    {std::cout  <<"Warning: Observations or ponderation vector is empty.\n";}
    for (unsigned int i(0) ; i<aObs->size();i++)
    {
      pObs.push_back(std::make_pair(&aPond->at(i),&aObs->at(i)));
    }
}

cLSQ_2dline::cLSQ_2dline(std::map<int,Pt2dr> * aObsMap,std::map<int,double> * aPondMap, ModelForm model)
{
    for (std::map<int, Pt2dr>::iterator it=aObsMap->begin(); it!=aObsMap->end(); ++it)
    {
        pObs.push_back(std::make_pair(&aPondMap->at(it->first),&it->second));
    }
}


cLSQ_2dline::cLSQ_2dline():mPond(1),mOk(0)
{

}

cLSQ_2dline::~cLSQ_2dline()
{
   // les observation ne sont pas detruite  car elle sont encore utilisées en dehors de cet objet
   //delete mObs;
}

void cLSQ_2dline::adjustModelL2()
{
    // Create L2SysSurResol to solve least square equation with 2 unknowns
    L2SysSurResol aSys(2);

    //For Each radiometric Couples, add the observations
    for(auto & pair_pointers : pObs){

        double aFormLin[2]={pair_pointers.second->x,1}; // b*x+a
        double poids=*pair_pointers.first;
        aSys.AddEquation(poids,aFormLin,pair_pointers.second->y);
    }

    Im1D_REAL8 aSol = aSys.GSSR_Solve(&mOk);

    if (mOk)
    {
        double* aData = aSol.data();
        //std::cout << "solution trouvée , b =" << aData[0] << " and a " << aData[1] << " \n";
        mModel=c2DLineModel(aData[1],aData[0]);
    } else {
        std::cout << "adjustment of a 2D line by LSQ L2 failed\n";
    }
}


void cLSQ_2dline::adjustModelL1()
{
    SystLinSurResolu aSys(2,pObs.size());

    //For Each radiometric Couples, add the observations

    for(auto & pair_pointers : pObs){

        double aFormLin[2]={pair_pointers.second->x,1}; // b*x+a
        int poids=*pair_pointers.first;
        //std::cout << "add equation to systlinsurREsolu : " << pair_pointers.second->x << " + a = " <<pair_pointers.second->y << ", weighting of " << poids << "\n";
        aSys.PushEquation(aFormLin,pair_pointers.second->y,poids);
    }

    Im1D_REAL8 aSol = aSys.L1Solve();
    mOk=1;

    if (mOk)
    {
        double* aData = aSol.data();
        //std::cout << "solution trouvée , b =" << aData[0] << " and a " << aData[1] << " \n";
        mModel=c2DLineModel(aData[1],aData[0]);
    } else {
        std::cout << "adjustment of a 2D line by LSQ L1 failed\n";
    }

}


void cLSQ_2dline::affiche()
{
    // affiche les résumés du modèle
    std::cout << "Modele ligne 2D LSQ - summary \n";
    // test si l'ajustement a déjà eu lieu
    if(mOk)
    {
    std::cout << "Observations :" << pObs.size() << "\n";
    std::cout << "a + b * x = y     a=" << mModel.getA() << ", b=" << mModel.getB() << ".\n";
    } else std::cout << "l'ajustement as échoué ou n'as pas encore été effectué.\n";
}
