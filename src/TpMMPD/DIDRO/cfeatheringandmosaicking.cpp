#include "cfeatheringandmosaicking.h"



cFeatheringAndMosaicOrtho::cFeatheringAndMosaicOrtho(int argc,char ** argv):lut_w(Im1D_REAL4(1,1))
{
    mDist=100; // distance chamfer de 100 pour le feathering/estompage
    mLambda=0.2;
    mDebug=1;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()   << EAMC(mFullDir,"ortho pattern", eSAM_IsPatFile)
                ,
                LArgMain()  << EAM(mNameMosaicOut,"Out",true, "Name of resulting map")
                << EAM(mDist,"Dist",true, "Distance for seamline feathering blending" )
                << EAM(mLambda,"Lambda",true, "lambda value for gaussian distance weighting, def 0.2" )
                << EAM(mDebug,"Debug",true, "Write intermediate results for debug purpose" )

                );

    if (!MMVisualMode)
    {

        mDir="./";
        mNameMosaicOut="mosaicFeatheringTest1.tif";
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
        mLFile = mICNM->StdGetListOfFile(mFullDir);

        cFileOriMnt MTD = StdGetFromPCP("MTDOrtho.xml",FileOriMnt);
        Box2dr boxMosaic(Pt2dr(MTD.OriginePlani().x,MTD.OriginePlani().y+MTD.ResolutionPlani().y*MTD.NombrePixels().y),Pt2dr(MTD.OriginePlani().x+MTD.ResolutionPlani().x*MTD.NombrePixels().x,MTD.OriginePlani().y));
        sz= Pt2di(MTD.NombrePixels().x,MTD.NombrePixels().y);
        aCorner=Pt2dr(boxMosaic._p0.x,boxMosaic._p1.y); // xmin, ymax;

        mosaic=Im2D_REAL4(sz.x,sz.y);
        NbIm=Im2D_REAL4(sz.x,sz.y);
        label=Im2D_U_INT1::FromFileStd("Label.tif");

        mSumDistExt=Im2D_INT4(sz.x,sz.y);
        mSumDistInter= Im2D_INT4(sz.x,sz.y);
        PondInterne=Im2D_REAL4(sz.x,sz.y);
        mSumWeighting=Im2D_REAL4(sz.x,sz.y);
        std::string aName;

        // look up table of weight
        lut_w=Im1D_REAL4(mDist+1);
        ELISE_COPY
                (
                    lut_w.all_pts(),
                    pow(0.5,pow((FX/((double)mDist-FX+1)),2*mLambda)),
                    lut_w.out()
                    );
        // replace by 1 distance over the threshold of mDist
        ELISE_COPY(select(lut_w.all_pts(),lut_w.in()>1),1,lut_w.out());

         if (mDebug) {
        std::cout << "Look up table of weight depending on the distance from seamline\n";
        for (unsigned int i(0); i<mDist;i++)
        {
            std::cout << "Feathering weighting at Chamfer Dist=" << i << " is equal to " << lut_w.At(i) << "\n";
        }
        }


        // compute chamfer distance from seamline and determine the number of images for each xy location
        std::cout << "Compute Chamfer Distance from seamline for each ortho\n";
        ChamferDist4AllOrt();

        if (mDebug)
        {
            // save the map of number of image
            aName="TestNbIm.tif";
            std::cout << "Save map of number of ortho used for each location , file " << aName << "\n";
            SaveTiff(aName,& NbIm);
            aName="TestSumDistInterne.tif";
            std::cout << "Save map of sum of chamfer distance inside enveloppe for each location , file " << aName << "\n";
            SaveTiff(aName,& mSumDistInter);
            aName="TestSumDistExt.tif";
            std::cout << "Save map of sum of chamfer distance outside enveloppe for each location , file " << aName << "\n";
            SaveTiff(aName,& mSumDistExt);
        }

        std::cout << "Compute weighting for each ortho\n";
        // compute the weighting for area where number of images is 1 or 2
        WeightingNbIm1and2();

        // Particular attention for area of 3 images overlapping.
        std::cout << "start monitoring of areas of 3 ortho blending\n";
        WeightingNbIm3();

        if (mDebug)
        {
            // save the map of number of images
            aName="TestPondInterne.tif";
            std::cout << "Save map of number of ortho used for each location , file " << aName << "\n";
            SaveTiff(aName,& PondInterne);
        }

        std::cout << "Compute mosaic by multipling orthoimage value by weighting map.\n";
        ComputeMosaic();

        if (mDebug)
        {
            aName="TestSumWeighting.tif";
            std::cout << "Save map of sum of weighting, should be equal to 1 everywhere , file " << aName << "\n";
            SaveTiff(aName,& mSumWeighting);
        }

        aName="TestMosaic.tif";
        SaveTiff(aName, & mosaic);

    }
}




// compute chamfer distance from border of the image, i do not like that at all. this function add a border prior to compute chanfer distance and remove it afterward
void cFeatheringAndMosaicOrtho::ChamferNoBorder(Im2D<U_INT1,INT> i2d) const
{
    int border(200);
    Im2D_U_INT1 tmp(i2d.sz().x+2*border,i2d.sz().y+2*border,1);
    ELISE_COPY(select(tmp.all_pts(),trans(i2d.in(1),-Pt2di(border,border))==0),0,tmp.out());
    Chamfer::d32.im_dist(tmp);
    ELISE_COPY(i2d.all_pts(),trans(tmp.in(255),Pt2di(border,border)),i2d.oclip());
}


void cFeatheringAndMosaicOrtho::ChamferDist4AllOrt()
{
    unsigned int i(0); // i is the label of the image - and the key
    for (auto &im : mLFile)
    {
        if (mDebug) std::cout << "Image num " << i << " is " << im <<" : loading and computing feathering buffer.\n";
        // open orthos
        mIms[i]= new cImGeo(mDir+im);
        mIm2Ds[i]=  mIms[i]->toRAM();
        mChamferDist[i]=Im2D_INT2(mIms[i]->Im().sz().x,mIms[i]->Im().sz().y,1);
        Pt2di sz(mIms[i]->Im().sz());
        Pt2di tr= -mIms[i]->computeTrans(aCorner);

        // la fonction chamfer fonctionne sur une image binaire et va calculer les distance à partir des pixels qui ont une valeur de 0.
        // la distance maximale est de 255

        //detect seam line for this image
        //1) translation of label to be in ortho geometry and set value to 0 for points that are not inside area of mosaicking for this image
        Im2D_U_INT1 tmp(sz.x,sz.y,1);
        ELISE_COPY(select(tmp.all_pts(), trans(label.in(),tr)!=(int)i),
                   //trans(label.in(),tr),
                   0,
                   tmp.out()
                   );

        // remove very small patch for wich we do not want to perform feathering because it is ugly otherwise
        Im2D_U_INT1 Ibin(tmp.sz().x,tmp.sz().y,0);
        ELISE_COPY(Ibin.all_pts(),tmp.in(),Ibin.out());
        // completely stupid but i have to ensure the border of the bin image is not ==1 otherwise I got the error out of bitmap in dilate spec Image
        // so the code may suffer weakness if a small patch is located near the border--ask marc
        ELISE_COPY(Ibin.border(2),0,Ibin.out());
        U_INT1 ** d = Ibin.data();
        Neighbourhood V8=Neighbourhood::v8();

        for (INT x=0; x < Ibin.sz().x; x++)
        {
            for (INT y=0; y < Ibin.sz().y; y++)
            {
                if (d[y][x] == 1)
                {
                    Liste_Pts_INT2 cc(2);
                    ELISE_COPY
                            (
                                // flux: composantes connexes du point.
                                conc
                                (
                                    Pt2di(x,y),
                                    Ibin.neigh_test_and_set(V8, 1, 0,  10) ), // on change la valeur des points sélectionné comme ça à la prochaine itération on ne sélectionne plus cette zone de composante connexe
                                2, // valeur bidonne, c'est juste le flux que je sauve dans cc
                                cc
                                );
                    // remove the patch
                    if (cc.card()<pow(mDist,2)) ELISE_COPY(cc.all_pts(),0,tmp.out());
                }
            }
        }

        // compute chamfer Distance d32
        ChamferNoBorder(tmp);

        // inverse value of distance because this is inside the enveloppe
        ELISE_COPY(mChamferDist[i].all_pts(),-tmp.in(),mChamferDist[i].out());

        // Initialise tmp again
        ELISE_COPY(tmp.all_pts(),1,tmp.out());
        ELISE_COPY(
                    select(mChamferDist[i].all_pts(),mChamferDist[i].in()==-2),// distance ==-2 are pixels on the seamline (more or less)
                    0,
                    tmp.out());
        // chamfer distance outside the enveloppe
        ChamferNoBorder(tmp);

        ELISE_COPY(
                    select(mChamferDist[i].all_pts(),mChamferDist[i].in()==0),
                    tmp.in(),
                    mChamferDist[i].out());

        // apply the hidden part masq

        //std::string aNameIm =  mICNM->Assoc2To1("Key-Assoc-OpIm2PC@",im,true).first;
        std::string aNamePC = "PC"+ im.substr(3,im.size()-2);
        Im2D_U_INT1 masq=Im2D_U_INT1::FromFileStd(aNamePC);

        // apply the mask of the ortho and the mask of the label
        ELISE_COPY(
                    select(mChamferDist[i].all_pts(),masq.in()==255) || select(mChamferDist[i].all_pts(), trans(label.in(),tr)==255),
                    255,
                    mChamferDist[i].out());

        // save chamfer map for checking purpose
        if (mDebug)
        {
            std::string aNameTmp="TestChamfer" + std::to_string(i) + ".tif";
            SaveTiff(aNameTmp, &mChamferDist[i]);
        }

        // comptage du nombre d'image a utiliser pour le blending (geométrie mosaic)
        ELISE_COPY(select(NbIm.all_pts(),trans(mChamferDist[i].in(mDist+1),-tr)<=mDist),
                   NbIm.in(0)+1,
                   NbIm.out()
                   );

        // Q? pourquoi je ne peux pas renseigner juste in() sans avoir une erreur genre  BITMAP :  out of domain while reading (RLE mode)

        // somme des distances de chamber dans les enveloppes externes  - pour gérer les cas de blending de 3 images
        ELISE_COPY(select(mSumDistExt.all_pts(),trans(mChamferDist[i].in_proj(),-tr)<=mDist & trans(mChamferDist[i].in_proj(),-tr)>0),
                   mSumDistExt.in(0)+trans(mChamferDist[i].in(0),-tr),
                   mSumDistExt.out()
                   );
        // somme des distances de chamber dans les enveloppes inter  - également pour gérer les cas de blending de 3 images
        ELISE_COPY(select(mSumDistInter.all_pts(),trans(mChamferDist[i].in_proj(),-tr)>=-mDist & trans(mChamferDist[i].in_proj(),-tr)<0),
                   mSumDistInter.in(0)+trans(mChamferDist[i].in(0),-tr),
                   mSumDistInter.out()
                   );

        i++;
    }

}

void cFeatheringAndMosaicOrtho::WeightingNbIm1and2()
{

    //  pondération contribution de l'image à l'intérieur de son enveloppe; je peux effectuer le calcul du facteur de pondération pour toutes les images
    //  partie fixe pondérée seulement par le nombre d'image

    ELISE_COPY(select(PondInterne.all_pts(), NbIm.in()==1 | NbIm.in()==2),
               1-(NbIm.in()-1)/NbIm.in(0),
               PondInterne.out()
               );

    // featherling dans l'enveloppe interne
    ELISE_COPY(select(PondInterne.all_pts(), (NbIm.in()==1 | NbIm.in()==2) && mSumDistInter.in() <=0 && mSumDistInter.in()>=-mDist),
               PondInterne.in()+ (1-(1/NbIm.in())) *  lut_w.in()[mDist+mSumDistInter.in()],
            PondInterne.out()
            );


    for (unsigned int i(0);  i < mIms.size();i++)
    {

        if (mDebug) std::cout << "Image" << i << ", computing weighting in overlap NbIm=1 and 2\n";

        Pt2di tr= mIms[i]->computeTrans(aCorner);
        mImWs[i]=Im2D_REAL4(mIms[i]->Im().sz().x,mIms[i]->Im().sz().y);
        // initialize to 0 because bug may appears otherwise and I do not like bugs
        ELISE_COPY(mImWs[i].all_pts(),0,mImWs[i].out());

        // internal enveloppe
        ELISE_COPY(select(mImWs[i].all_pts(),mChamferDist[i].in(0)<0),
                   trans(PondInterne.in(0),-tr),
                   mImWs[i].out()
                   );
        // external enveloppe
        ELISE_COPY(select(mImWs[i].all_pts(),mChamferDist[i].in(0)>0  & mChamferDist[i].in(0)<=mDist & (trans(NbIm.in(),-tr)==1 |trans(NbIm.in(),-tr)==2)),
                   1-trans(PondInterne.in(0),-tr),
                   mImWs[i].out()
                   );


    }
}

void cFeatheringAndMosaicOrtho::WeightingNbIm3()
{

    Im2D_U_INT1 Ibin(sz.x,sz.y,0);
    ELISE_COPY(select(Ibin.all_pts(),NbIm.in()==3), 1,Ibin.out());
    U_INT1 ** d = Ibin.data();
    int count(0); // counter for number of area of 3 images overlapping
    Neighbourhood V8=Neighbourhood::v8();

    for (INT x=0; x < Ibin.sz().x; x++)
    {
        for (INT y=0; y < Ibin.sz().y; y++)
        {
            if (d[y][x] == 1)
            {
                count++;
                Liste_Pts_INT2 cc(2);
                ELISE_COPY
                        (
                            // flux: composantes connexes du point.
                            conc
                            (
                                Pt2di(x,y),
                                Ibin.neigh_test_and_set(V8, 1, 0,  20) ), // on change la valeur des points sélectionné comme ça à la prochaine itération on ne sélectionne plus cette zone de composante connexe
                            1, // valeur bidonne, c'est juste le flux que je sauve dans cc
                            cc
                            );

                // determine la boite englobante
                Pt2di pmax,pmin;
                // temporary map enabling a quick selection of these pixels
                Im2D_INT2    CurrentArea(sz.x,sz.y,0);
                ELISE_COPY
                        (
                            cc.all_pts(),
                            Virgule(FX,FY),
                            ((pmax.VMax())|(pmin.VMin())) );

                ELISE_COPY
                        (
                            cc.all_pts(),
                            1,
                            CurrentArea.out());

                unsigned int val;
                std::vector<int> labs;
                for (unsigned int x2(0);x2<cc.image().sz().x;x2++){
                    // cumbersome but only way I found to retrieve point position from list of points
                    Pt2di pt(cc.image().GetR(Pt2di(x2,0)),cc.image().GetR(Pt2di(x2,1)));
                    val = label.GetR(pt);
                    //test if value is in the labs array
                    if (std::find(std::begin(labs), std::end(labs), val) == std::end(labs) && val!=255)
                    {labs.push_back(val);}
                    // std::cout << "add label image " << val << "\n"       ;}
                }
                std::cout << "I got the 3 images for an ovelap area, images labels are " << labs <<"\n";

                // load 3 chamfer distance for ease of manipulation
                Pt2di beSz(pmax.x-pmin.x,pmax.y-pmin.y);
                std::cout << "size of the area : " << beSz << ", location on the global orthophoto : " << pmin << " and " << pmax <<"\n";
                Im2D_INT2 dist1(beSz.x,beSz.y),dist2(beSz.x,beSz.y),dist3(beSz.x,beSz.y);
                Im2D_REAL4 gateau12(beSz.x,beSz.y);
                Pt2di tr0 = mIms[labs[0]]->computeTrans(aCorner)+pmin;// double translation, une vers le système global, une vers la boite englobante de la zone
                ELISE_COPY(dist1.all_pts(),trans(mChamferDist[labs[0]].in(0),tr0),dist1.out());
                Pt2di tr1 = mIms[labs[1]]->computeTrans(aCorner)+pmin;
                ELISE_COPY(dist2.all_pts(),trans(mChamferDist[labs[1]].in(0),tr1),dist2.out());
                Pt2di tr2 = mIms[labs[2]]->computeTrans(aCorner)+pmin;
                ELISE_COPY(dist2.all_pts(),trans(mChamferDist[labs[2]].in(0),tr2),dist3.out());

                // strategy; firt im1 and im2 are blended, then im3 with the blend of im1 im2. but in fact, process first im3


                // feathering dans l'enveloppe interne de l'image numéro 3
                Pt2di tr = mIms[labs[2]]->computeTrans(aCorner);
                ELISE_COPY(select(mImWs[labs[2]].all_pts(),mChamferDist[labs[2]].in()<0 & trans(CurrentArea.in(0),-tr)==1),
                           0.5+ 0.5 * lut_w.in()[mDist+mChamferDist[labs[2]].in()],
                            mImWs[labs[2]].out()
                            );
                // feathering dans l'enveloppe externe de l'image numéro 3
                ELISE_COPY(select(mImWs[labs[2]].all_pts(),mChamferDist[labs[2]].in()>0 & mChamferDist[labs[2]].in()<mDist & trans(CurrentArea.in(0),-tr)==1),
                           1-(0.5+ 0.5 * lut_w.in()[mDist-mChamferDist[labs[2]].in()]),
                            mImWs[labs[2]].out()
                            );
                // copie de ceci dans PondInterne
                ELISE_COPY(select(PondInterne.all_pts(), CurrentArea.in()==1 & trans(mChamferDist[labs[2]].in(0),tr)<0),
                           trans(mImWs[labs[2]].in(0),tr),
                           PondInterne.out()
                            );

                // copie également dans gateau12, part to divide between 1 and 2
              //  ELISE_COPY(select(gateau12.all_pts(), trans(CurrentArea.in(),pmin)==1 & trans(mChamferDist[labs[2]].in(),tr2)<0),
                //           trans(mImWs[labs[2]].in(),tr2),
                 //          gateau12.out()
                 //           );
                ELISE_COPY(gateau12.all_pts(),0,gateau12.out());
                ELISE_COPY(select(gateau12.all_pts(), trans(CurrentArea.in(),pmin)==1), //& trans(mChamferDist[labs[2]].in(),tr2)>0),
                           1-trans(mImWs[labs[2]].in(0),tr2),
                           gateau12.out()
                            );

                // ok, im1 with im2 now
                //redefinition of a seamline between the two images, seamline on the enveloppe of the 3th image
                Im2D_U_INT1  tmp(beSz.x,beSz.y,1);
                Im2D_INT2    NewDist(beSz.x,beSz.y,1);
                ELISE_COPY(select(tmp.all_pts(),dist1.in()==dist2.in()) || select(tmp.all_pts(),dist1.in()==-2 & dist2.in()<5),
                           0,
                           tmp.out()
                           );
                ChamferNoBorder(tmp);

                ELISE_COPY(select(dist1.all_pts(),dist1.in()>dist2.in()),tmp.in(),NewDist.out());
                // warn I just changed value of dist1 so i cannot compare again. need for another tmp image
                ELISE_COPY(select(dist1.all_pts(),dist1.in()<=dist2.in()),-tmp.in(),NewDist.out());
                //ELISE_COPY(dist1.all_pts(),tmp2.in(),dist1.out());
                //ELISE_COPY(dist2.all_pts(),-tmp2.in(),dist2.out());
                // ok I got my new seamline between 1 and 2 and my new chamfer distance, negative inside envelope, positive outside

                // env interne im 1
                Im2D_REAL4 w0(beSz.x,beSz.y),w1(beSz.x,beSz.y);
                ELISE_COPY(w0.all_pts(),0,w0.out());
                ELISE_COPY(select(NewDist.all_pts(),NewDist.in()<=0 & NewDist.in()>-mDist),
                           gateau12.in() * (0.5+ 0.5*lut_w.in()[mDist+NewDist.in()]),
                           w0.out()
                            );
                ELISE_COPY(select(NewDist.all_pts(),NewDist.in()>0 & NewDist.in()<mDist),
                           gateau12.in() * (1 - 0.5-0.5*lut_w.in()[mDist-NewDist.in()]),
                           w0.out()
                            );
                ELISE_COPY(w1.all_pts(),gateau12.in()-w0.in(),w1.out());

                //copy these weighting in the mImWs

                ELISE_COPY( mImWs[labs[0]].all_pts(),
                            mImWs[labs[0]].in() + trans(w0.in(0),-mIms[labs[0]]->computeTrans(aCorner)-pmin),
                            mImWs[labs[0]].out()
                            );

                ELISE_COPY( mImWs[labs[1]].all_pts(),
                            mImWs[labs[1]].in() + trans(w1.in(0),-mIms[labs[1]]->computeTrans(aCorner)-pmin),
                            mImWs[labs[1]].out()
                            );

                   std::string aName;

                  aName="TestGateau12.tif";
                  SaveTiff(aName,& gateau12);
                  aName="TestDist1.tif";
                  SaveTiff(aName,& dist1);
                  aName="TestDist2.tif";
                  SaveTiff(aName,& dist2);
                  aName="TestW0.tif";
                  SaveTiff(aName,& w0);
                  aName="TestW1.tif";
                  SaveTiff(aName,& w1);
                  aName="TestNewDist.tif";
                  SaveTiff(aName,& NewDist);


            }
        }
    }


}



void cFeatheringAndMosaicOrtho::ComputeMosaic()
{
    for (unsigned int i(0);  i < mIms.size();i++)
    {
        Pt2di tr= mIms[i]->computeTrans(aCorner);
        // mosaic
        ELISE_COPY(select(mosaic.all_pts(),trans(mImWs[i].in(0),tr)>0),
                   mosaic.in()+trans(mIm2Ds[i].in()*mImWs[i].in(),tr) ,
                   mosaic.out()
                   );
        // sum of weighting, to check it is well equal to 1 everywhere
        if (mDebug)
        {
            ELISE_COPY(select(mSumWeighting.all_pts(),trans(mImWs[i].in(0),tr)>0),
                       mSumWeighting.in()+trans(mImWs[i].in(0),tr) ,
                       mSumWeighting.out()
                       );
            // individual weighting
            std::string aNameTmp="TestWeighting" + std::to_string(i) + ".tif";
            SaveTiff(aNameTmp, &mImWs[i]);
        }
    }
}





template <class T,class TB> void  cFeatheringAndMosaicOrtho::SaveTiff(std::string & aName, Im2D<T,TB> * aIm)
{
    Tiff_Im  aTF
            (
                aName.c_str(),
                aIm->sz(),
                GenIm::real4,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
                );
    ELISE_COPY(aIm->all_pts(),aIm->in(),aTF.out());
}
