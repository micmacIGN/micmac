#include "TaskCorrel.h"

//  ============================= **************** =============================
//  *                             cTriForTiepTri                               *
//  ============================= **************** =============================
cTriForTiepTri::cTriForTiepTri(cAppliTaskCorrel *aAppli, cTri3D aTri3d, int & ind):
    mNumImg (-1),
    mPt1    (Pt2dr(0.0,0.0)),
    mPt2    (Pt2dr(0.0,0.0)),
    mPt3    (Pt2dr(0.0,0.0)),
    mAppli  (aAppli),
    mTri3D_  (aTri3d),
    mrprjOK (false),
    mInd    (ind)
{}

//  ============================= **************** =============================
//  *                             reprj                                        *
//  ============================= **************** =============================
bool cTriForTiepTri::reprj(cImgForTiepTri * aImg)
{
    mNumImg = aImg->Num();
    Pt3dr Pt1 = mTri3D_.P1();
    Pt3dr Pt2 = mTri3D_.P2();
    Pt3dr Pt3 = mTri3D_.P3();
    if      (
                  aImg->CamGen()->PIsVisibleInImage(Pt1)
              &&  aImg->CamGen()->PIsVisibleInImage(Pt2)
              &&  aImg->CamGen()->PIsVisibleInImage(Pt3)
            )
    {
        mrprjOK = true;
        mPt1 = aImg->CamGen()->Ter2Capteur(Pt1);
        mPt2 = aImg->CamGen()->Ter2Capteur(Pt2);
        mPt3 = aImg->CamGen()->Ter2Capteur(Pt3);
        return true;
    }
    else
        {
            mrprjOK = false;
            return false;
        }
}
//  ============================= **************** =============================
//  *                             reprj_pure                                   *
//  ============================= **************** =============================
bool cTriForTiepTri::reprj_pure(cImgForTiepTri * aImg, Pt2dr & P1, Pt2dr & P2, Pt2dr & P3)
{
    Pt3dr Pt1 = mTri3D_.P1();
    Pt3dr Pt2 = mTri3D_.P2();
    Pt3dr Pt3 = mTri3D_.P3();
    if      (
                  aImg->CamGen()->PIsVisibleInImage(Pt1)
              &&  aImg->CamGen()->PIsVisibleInImage(Pt2)
              &&  aImg->CamGen()->PIsVisibleInImage(Pt3)
            )
    {
        P1 = aImg->CamGen()->Ter2Capteur(Pt1);
        P2 = aImg->CamGen()->Ter2Capteur(Pt2);
        P3 = aImg->CamGen()->Ter2Capteur(Pt3);
        return true;
    }
    else
        {
            return false;
        }
}


//  ============================= **************** =============================
//  *                             isCollinear                                  *
//  ============================= **************** =============================
bool isCollinear(Pt3dr & P1, Pt3dr & P2, Pt3dr & P3)
{
  Pt3dr u = P2-P1; //u
  Pt3dr v = P3-P1; //v
  /*
  Pt3dr croosPrd(u.x*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.x-v.x);
  if (croosPrd.x < DBL_EPSILON && croosPrd.y < DBL_EPSILON && croosPrd.z < DBL_EPSILON)
     return true;
     */
  Pt3dr croosPrd(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y-v.x);
  if (croosPrd.x == 0 && croosPrd.y == 0 && croosPrd.z == 0)
     return true;
  else
     return false;
}

//  ============================= **************** =============================
//  *                             valElipse                                    *
//  ============================= **************** =============================
double cTriForTiepTri::valElipse(int & aNInter)
{
    if (!mrprjOK || mNumImg == -1)
    {
        if (aNInter!=0) {cout<<"Projection error !"<<endl;}
        return -1.0;
    }
    else
    {
        double aSurf =  (mPt1-mPt2) ^ (mPt1-mPt3);
        if(this->mAppli->ZBuf_InverseOrder())
        {
            aSurf=-aSurf;
        }
        //cout<<"SUFFFFFFFF "<<aSurf<<endl;
        Pt3dr Pt1 = mTri3D_.P1();
        Pt3dr Pt2 = mTri3D_.P2();
        Pt3dr Pt3 = mTri3D_.P3();
        bool isColnr = isCollinear(Pt1, Pt2, Pt3);  //si 3 point du triangle sont collinears
        if (-aSurf > mAppli->SEUIL_SURF_TRIANGLE() && mrprjOK && !isColnr)
        {
        //creer plan 3D local contient triangle
            cElPlan3D * aPlanLoc = new cElPlan3D(Pt1, Pt2, Pt3);    //ATTENTION : peut causer error vuunit si 3 point collinear
            ElRotation3D aRot_PE = aPlanLoc->CoordPlan2Euclid();
            ElRotation3D aRot_EP = aRot_PE.inv();
	    //calcul coordonne sommet triangle dans plan 3D local (devrait avoir meme Z)
            Pt3dr aPtP0 = aRot_EP.ImAff(Pt1); //sommet triangle on plan local
            Pt3dr aPtP1 = aRot_EP.ImAff(Pt2);
            Pt3dr aPtP2 = aRot_EP.ImAff(Pt3);
	    //creer translation entre coordonne image global -> coordonne image local du triangle (plan image)
            ElAffin2D aAffImG2ImL(ElAffin2D::trans(mPt1));
            Pt2dr aPtPIm0 = aAffImG2ImL(mPt1);
            Pt2dr aPtPIm1 = aAffImG2ImL(mPt2);
            Pt2dr aPtPIm2 = aAffImG2ImL(mPt3);
	    //calcul affine entre plan 3D local (elimine Z) et plan 2D local
            ElAffin2D aAffLc2Im;
            aAffLc2Im = aAffLc2Im.FromTri2Tri(  Pt2dr(aPtP0.x, aPtP0.y),
                                                Pt2dr(aPtP1.x, aPtP1.y),
                                                Pt2dr(aPtP2.x, aPtP2.y),
                                                aPtPIm0,aPtPIm1,aPtPIm2
                                             );
        //calcul vector max min pour choisir img master
            // double vecA_cr =  aAffLc2Im.I10().x*aAffLc2Im.I10().x + aAffLc2Im.I10().y*aAffLc2Im.I10().y;
            double vecA_cr =  square_euclid(aAffLc2Im.I10());
            //  double vecB_cr =  aAffLc2Im.I01().x*aAffLc2Im.I01().x + aAffLc2Im.I01().y*aAffLc2Im.I01().y;
            double vecB_cr = square_euclid(aAffLc2Im.I01());


            //  double AB_cr   =  pow(aAffLc2Im.I10().x*aAffLc2Im.I01().x,2) + pow(aAffLc2Im.I10().y*aAffLc2Im.I01().y,2);

             double AB_cr =  ElSquare(scal(aAffLc2Im.I10(),aAffLc2Im.I01()));


	    //double theta_max =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(0.5);
            //Interaction : disp ellipse on image:
            if (aNInter > 1)
            {
                //calcul le cercle discretize dans le plan 3D local
                Video_Win * aVW = mAppli->VVW()[mNumImg];
                double rho;
                double aSclElps=-1;
                if (aSclElps == -1)
                {
                    double rho1 = sqrt(aPtP1.x*aPtP1.x + aPtP1.y*aPtP1.y);
                    double rho2 = sqrt(aPtP2.x*aPtP2.x + aPtP2.y*aPtP2.y);
                    if (rho1 > rho2)
                        rho = rho1;
                    else
                        rho = rho2;

                }
                else
                {
                    double scale = euclid ( aAffLc2Im.inv()(Pt2dr(0,0)) - aAffLc2Im.inv()(Pt2dr(aSclElps,0)) );
                    rho = scale;
                }
                double aNbPt = 100;
                vector<Pt2dr> VCl;
                vector<Pt2dr> VTri;VTri.push_back(mPt1);VTri.push_back(mPt2);VTri.push_back(mPt3);
                for (uint aKP=0; aKP<aNbPt; aKP++)
                {
                    Pt2dr ptCrlImg;
                    ptCrlImg = aAffImG2ImL.inv()(aAffLc2Im(Pt2dr::FromPolar(rho, aKP*2*PI/aNbPt)));
                    VCl.push_back(ptCrlImg);
                }
                Line_St lstLineG(aVW->pdisc()(P8COL::green),1);
                ELISE_COPY(aVW->all_pts(), mAppli->VImgs()[mNumImg]->Tif().in_proj(), aVW->ogray());
                aVW->draw_poly_ferm(VCl, lstLineG);
                aVW->draw_poly_ferm(VTri, aVW->pdisc()(P8COL::red));
                if (mNumImg == int(mAppli->VVW().size() - 1))
                    aVW->clik_in();
            }

            double aDisc = ElSquare(vecA_cr - vecB_cr) + 4*AB_cr;

/*
std::cout <<"Discr = " << aDisc << "\n";
if (aDisc <0)
{
    std::cout << "=====================\n";
    getchar();
}
*/

            return (vecA_cr + vecB_cr - sqrt(aDisc))  / 2.0 ;
        }
        else
        {
            if (isColnr && aNInter!=0)
                {
                    cout<<Pt1<<Pt2<<Pt3<<" => 3 Pts tri collinear !"<<endl;
                }
            if (-aSurf < mAppli->SEUIL_SURF_TRIANGLE() && aNInter!=0)
                {
                    cout<<Pt1<<Pt2<<Pt3<<"Surface tri too small !"<<endl;
                }
            mrprjOK = false;
            return -1.0;
        }
    }
}


