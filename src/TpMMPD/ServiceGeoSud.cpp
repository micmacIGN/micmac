#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"

#include "Surf.h"

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic push
    #endif
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

Pt2di getImageSize(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=aName.size()-1;(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std::auto_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName));
        if (aRes.get())
        {
            return Std2Elise(aRes->Sz(1));
        }
    }
#endif

    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    return aTif.sz();
}

template <class Type,class TyBase>
TIm2D<Type,TyBase>* createTIm2DFromFile(std::string const &aName, Pt2di const &PminCrop,Pt2di const &SzCrop)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=aName.size()-1;(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std::auto_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName));
        if (aRes.get()!=NULL)
        {
            std::auto_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(SzCrop));
            aRes->LoadCanalCorrel(sLowLevelIm<Type>
                                  (
                                   anTIm2D->_the_im.data_lin(),
                                   anTIm2D->_the_im.data(),
                                   Elise2Std(anTIm2D->sz())
                                   ),
                                  1,//deZoom
                                  cInterfModuleImageLoader::tPInt(0,0),//aP0Im
                                  cInterfModuleImageLoader::tPInt(PminCrop.x,PminCrop.y),//aP0File
                                  cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y));
            return anTIm2D.release();
        }
    }
#endif

    std::auto_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(SzCrop));
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    ELISE_COPY(anTIm2D->_the_im.all_pts(),trans(aTif.in(),PminCrop),anTIm2D->_the_im.out());
    return anTIm2D.release();
}


template <class Type,class TyBase>
TIm2D<Type,TyBase>* createTIm2DFromFile(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=aName.size()-1;(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        std::auto_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName));
        if (aRes.get()!=NULL)
        {
            Pt2di aSz = Std2Elise(aRes->Sz(1));
            std::auto_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(aSz));
            aRes->LoadCanalCorrel(sLowLevelIm<Type>
                                   (
                                    anTIm2D->_the_im.data_lin(),
                                    anTIm2D->_the_im.data(),
                                    Elise2Std(anTIm2D->sz())
                                    ),
                                   1,
                                   cInterfModuleImageLoader::tPInt(0,0),
                                   cInterfModuleImageLoader::tPInt(0,0),
                                   cInterfModuleImageLoader::tPInt(aSz.x,aSz.y));
            return anTIm2D.release();
        }
    }
#endif
    return new TIm2D<Type,TyBase>(Im2D<Type,TyBase>::FromFileStd(aName));
}

double correl(std::vector<double> const &f1,std::vector<double> const &f2)
{
    std::vector<double>::const_iterator it1,it2,fin1,fin2;
    it1=f1.begin();
    it2=f2.begin();
    fin1=f1.end();
    fin2=f2.end();

    double sa=0.,sa2=0.,sb=0.,sb2=0,sab=0.;
    int N=0;
    for(;(it1!=fin1)&&(it2!=fin2);)
    {
        sa += (*it1);
        sa2 += pow((*it1),2);
        sb += (*it2);
        sb2 += pow((*it2),2);
        sab += (*it1)*(*it2);
        ++N;
        ++it1;
        ++it2;
    }
    double pdtVar = (sa2 - pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar!=0.)
        return (sab-sa*sb/N)/sqrt(pdtVar);
    return 0.;
}

double correl(std::vector<double> const &f1,std::vector<double> const &f2,double NoData)
{
    std::vector<double>::const_iterator it1,it2,fin1,fin2;
    it1=f1.begin();
    it2=f2.begin();
    fin1=f1.end();
    fin2=f2.end();

    double sa=0.,sa2=0.,sb=0.,sb2=0,sab=0.;
    int N=0;
    for(;(it1!=fin1)&&(it2!=fin2);)
    {
        if (((*it1)!=NoData)&&((*it2)!=NoData))
        {
            sa += (*it1);
            sa2 += pow((*it1),2);
            sb += (*it2);
            sb2 += pow((*it2),2);
            sab += (*it1)*(*it2);
            ++N;
        }
        ++it1;
        ++it2;
    }
    if (N == 0)
        return 0.;
    double pdtVar = (sa2 - pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar==0.)
        return 0.;
    return (sab-sa*sb/N)/sqrt(pdtVar);
}


double dist(DigeoPoint const &pt1, DigeoPoint const &pt2)
{
    double dmin=0.;
    bool first = true;

    for (size_t i=0;i<pt1.entries.size();++i)
    {
        DigeoPoint::Entry const &entrie1 = pt1.entries[i];
        for (size_t j=0;j<pt2.entries.size();++j)
        {
            DigeoPoint::Entry const &entrie2 = pt2.entries[j];
            double d = 0.;
            for( int k=0;k<DIGEO_DESCRIPTOR_SIZE;++k)
            {
                d += pow(entrie1.descriptor[k] - entrie2.descriptor[k], 2);
            }
            if (first)
            {
                dmin = d;
            }
            else
            {
                first = false;
                if (dmin>d)
                {
                    dmin = d;
                }
            }
        }
    }
    return dmin;
}

int TP2GCP(std::string const &aNameFileMNT,
               std::string const &aNameFileOrtho,
               std::string const &aNameFilePOIOrtho,
               std::string const &aNameFileGrid,
               std::string const &aNameFileImage,
               std::string const &aNameFilePOIImage,
               double seuilPixel,
               double seuilCorrel,
               int SzW,
               std::string const &aNameResult)
{
    int SzMaxImg = 4000 * 4000;

    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    std::auto_ptr<TIm2D<REAL4,REAL8> > aMntImg(createTIm2DFromFile<REAL4,REAL8>(aMntOri.NameFileMnt()));
    if (aMntImg.get()==NULL)
    {
        cerr << "Error in "<<aMntOri.NameFileMnt()<<std::endl;
        return EXIT_FAILURE;
    }

    // Chargement de l'Ortho
    cFileOriMnt aOrthoOri=  StdGetFromPCP(aNameFileOrtho,FileOriMnt);
    std::cout << "Taille de l'ortho : "<<aOrthoOri.NombrePixels().x<<" "<<aOrthoOri.NombrePixels().y<<std::endl;
    std::auto_ptr<TIm2D<U_INT1,INT4> > OrthoImg(NULL);
    Pt2di OrthoSz = getImageSize(aOrthoOri.NameFileMnt());
    if (OrthoSz.x * OrthoSz.y < SzMaxImg)
    {
        OrthoImg.reset(createTIm2DFromFile<U_INT1,INT4>(aOrthoOri.NameFileMnt()));
        if (OrthoImg.get()==NULL)
        {
            cerr << "Error in "<<aOrthoOri.NameFileMnt()<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Chargement des points d'interet dans l'ortho
    vector<DigeoPoint> keypointsOrtho;

    if ( !DigeoPoint::readDigeoFile( aNameFilePOIOrtho, true, keypointsOrtho ) ){
        cerr << "WARNING: unable to read keypoints in [" << aNameFilePOIOrtho << "]" << endl;
        return EXIT_FAILURE;
    }
    std::cout << "Nombre de points dans l'ortho : "<<keypointsOrtho.size()<<std::endl;

    // Chargement des points d'interet dans l'image
    vector<DigeoPoint> keypointsImage;

    if ( !DigeoPoint::readDigeoFile( aNameFilePOIImage, true, keypointsImage ) ){
        cerr << "WARNING: unable to read keypoints in [" << aNameFilePOIImage << "]" << endl;
        return EXIT_FAILURE;
    }
    std::cout << "Nombre de points dans l'image : "<<keypointsImage.size()<<std::endl;

    // Chargement de la grille et de l'image
    std::auto_ptr<TIm2D<U_INT1,INT4> > img(NULL);
    Pt2di ImgSz = getImageSize(aNameFileImage);
    if (ImgSz.x * ImgSz.y < SzMaxImg)
    {
        img.reset(createTIm2DFromFile<U_INT1,INT4>(aNameFileImage));
        if (img.get()==NULL)
        {
            cerr << "Error in "<<aNameFileImage<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Chargement de la grille et de l'image
    ElAffin2D oriIntImaM2C;
    std::auto_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));

    bool verbose=true;

    vector<DigeoPoint>::const_iterator it,fin=keypointsOrtho.end();
    vector<DigeoPoint>::const_iterator it2,fin2=keypointsImage.end();

    double seuilPixel2 = pow(seuilPixel,2);

    std::ofstream fic(aNameResult.c_str());
    fic << std::setprecision(15);

    // Export sous forme de dico appuis et mesures
    cSetOfMesureAppuisFlottants aSetDicoMesure;

    cMesureAppuiFlottant1Im aDicoMesure;
    aDicoMesure.NameIm()=aNameFileImage;
    cDicoAppuisFlottant aDicoAppuis;
    int cpt = 0;

    if (verbose) std::cout << "aOrthoOri.ResolutionPlani() : "<<aOrthoOri.ResolutionPlani()<<" / aOrthoOri.OriginePlani() : "<<aOrthoOri.OriginePlani()<<std::endl;

    double NoData = -9999.;

    // on parcourt les points sift de l'ortho
    for(it=keypointsOrtho.begin();it!=fin;++it)
    {
        DigeoPoint const &ptSift = (*it);
        // On estime la position 3D du point
        // ToDo: ajouter l'interpolation de l'alti dans le mnt
        Pt2dr ptOrtho(ptSift.x,ptSift.y);
        //if (verbose) std::cout << "Point img dans l'ortho : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
        ptOrtho.x =  ptOrtho.x*aOrthoOri.ResolutionPlani().x + aOrthoOri.OriginePlani().x;
        ptOrtho.y =  ptOrtho.y*aOrthoOri.ResolutionPlani().y + aOrthoOri.OriginePlani().y;
        //if (verbose) std::cout << "Point terrain 2D : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
        // Position dans le MNT
        Pt2dr ptMnt;
        ptMnt.x = (ptOrtho.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
        ptMnt.y = (ptOrtho.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
        //if (verbose) std::cout << "Point img dans le Mnt : "<<ptMnt.x<<" "<<ptMnt.y<<std::endl;
        double alti = aMntImg->getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
        if (alti == NoData)
        {
            std::cout << "Pas d'altitude trouvee pour le point : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
            std::cout << "On passe au point suivant"<<std::endl;
            break;
        }
        //if (verbose) std::cout << "Altitude Mnt : "<<alti<<std::endl;
        // Position dans l'image
        Pt3dr pt3(ptOrtho.x,ptOrtho.y,alti);
        //if (verbose) std::cout << "Point terrain : "<<pt3.x<<" "<<pt3.y<<" "<<pt3.z<<std::endl;
        Pt2dr pImg = aCamera->R3toF2(pt3);
        //if (verbose) std::cout << "Point Image : "<<pImg.x<<" "<<pImg.y<<std::endl;


        std::vector<double> fenOrtho;
        for(int l=ptSift.y-SzW;l<=(ptSift.y+SzW);++l)
        {
            for(int c=ptSift.x-SzW;c<=(ptSift.x+SzW);++c)
            {
                fenOrtho.push_back(OrthoImg->getr(Pt2dr(c,l),NoData));
            }
        }


        DigeoPoint ptHomoDmin;
        Pt3dr pTerrainDmin;
        double dmin = -1.;
        // On cherche un POIImage proche de pImg
        for(it2=keypointsImage.begin();it2!=fin2;++it2)
        {
            DigeoPoint const &ptSift2 = (*it2);
            Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(ptSift2.x,ptSift2.y),pt3.z);

            double d2 = pow(pImg.x-ptSift2.x,2) + pow(pImg.y-ptSift2.y,2);
            if (d2<seuilPixel2)
            {
                double d = dist(ptSift,ptSift2);
                if ((dmin<0)||(dmin>d))
                {
                    dmin = d;
                    ptHomoDmin = ptSift2;
                    pTerrainDmin = pTerrain;
                }
            }
        }
        if (dmin>=0)
        {
            // On valide le point avec la correlation

            std::vector<double> fenImage;

            if (img.get()!=NULL)
            {
                // version sans crop
                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt3dr P3D;
                        P3D.x = pTerrainDmin.x + dx * aOrthoOri.ResolutionPlani().x;
                        P3D.y = pTerrainDmin.y + dy * aOrthoOri.ResolutionPlani().y;
                        P3D.z = pTerrainDmin.z;

                        Pt2dr p2 = aCamera->R3toF2(P3D);
                        fenImage.push_back(img->getr(p2,NoData));
                    }
                }
            }
            else
            {
                // version avec crop
                std::vector<Pt2dr> vCoordImage;
                double cmin = 0 ,cmax = 0 ,lmin = 0,lmax = 0;
                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt3dr P3D;
                        P3D.x = pTerrainDmin.x + dx * aOrthoOri.ResolutionPlani().x;
                        P3D.y = pTerrainDmin.y + dy * aOrthoOri.ResolutionPlani().y;
                        P3D.z = pTerrainDmin.z;

                        Pt2dr p2 = aCamera->R3toF2(P3D);
                        vCoordImage.push_back(p2);
                        if (vCoordImage.size()==1)
                        {
                            cmin = p2.x;
                            cmax = p2.x;
                            lmin = p2.y;
                            lmax = p2.y;
                        }
                        else
                        {
                            if (cmin>p2.x)
                                cmin=p2.x;
                            else if (cmax<p2.x)
                                cmax=p2.x;
                            if (lmin>p2.y)
                                lmin=p2.y;
                            else if (lmax<p2.y)
                                lmax=p2.y;
                        }
                    }
                }
                Pt2di PminCrop((int)round_ni(cmin-1.),(int)round_ni(lmin-1.));
                Pt2di SzCrop((int)round_ni(cmax-PminCrop.x+2),(int)round_ni(lmax-PminCrop.y+2));
                //std::cout << "Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                std::auto_ptr<TIm2D<U_INT1,INT4> > cropImg(createTIm2DFromFile<U_INT1,INT4>(aNameFileImage,PminCrop,SzCrop));
                if (cropImg.get()==NULL)
                {
                    cerr << "Error in "<<aNameFileImage<<" Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                    return EXIT_FAILURE;
                }
                for(size_t i=0;i<vCoordImage.size();++i)
                {
                    fenImage.push_back(cropImg->getr(Pt2dr(vCoordImage[i].x-PminCrop.x,vCoordImage[i].y-PminCrop.y),NoData));
                }
            }

            // Coeff de correlation
            double coef = correl(fenOrtho,fenImage,NoData);
            if (coef >= seuilCorrel)
            {
                if (verbose) std::cout << "Point Dmin : "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<" distSift="<<dmin<<" Correl="<<coef<<std::endl;
                fic << ptSift.x<<" "<<ptSift.y<<" "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<std::endl;

                cOneMesureAF1I aMesure;
                cOneAppuisDAF aAppui;

                Pt2dr pt(ptSift.x,ptSift.y);
                aMesure.PtIm()=pt;

                aAppui.Pt() = pt3;
                aAppui.Incertitude()=Pt3dr(1,1,1);

                std::ostringstream oss;
                oss << "Point_"<<cpt;
                ++cpt;
                aAppui.NamePt()=oss.str();
                aMesure.NamePt()=oss.str();

                aDicoAppuis.OneAppuisDAF().push_back(aAppui);
                aDicoMesure.OneMesureAF1I().push_back(aMesure);
            }
        }

        // Autre approche: on teste tous le voisinage
/*
        double coefmax = -1.;
        Pt2dr ptMaxCorrel;
        for(int dl = -seuilPixel;dl<=seuilPixel;++dl)
        {
            for(int dc = -seuilPixel;dc<=seuilPixel;++dc)
            {
                Pt2dr pt2;
                pt2.x = pImg.x + dc;
                pt2.y = pImg.y + dl;
                Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(pt2.x,pt2.y),pt3.z);
                std::vector<double> fenImage;
                // Plutot que de reprojetter tous les points de la fenetre, on interpole la projection
                Pt3dr pTerrain2;
                pTerrain2.x = pTerrain.x + aOrthoOri.ResolutionPlani().x;
                pTerrain2.y = pTerrain.y;
                pTerrain2.z = pTerrain.z;
                Pt2dr ptImg2 = aCamera->R3toF2(pTerrain2);
                double deltax_col = ptImg2.x - pt2.x;
                double deltax_lig = ptImg2.y - pt2.y;

                pTerrain2.x = pTerrain.x;
                pTerrain2.y = pTerrain.y + aOrthoOri.ResolutionPlani().y;
                pTerrain2.z = pTerrain.z;
                ptImg2 = aCamera->R3toF2(pTerrain2);
                double deltay_col = ptImg2.x - pt2.x;
                double deltay_lig = ptImg2.y - pt2.y;

                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt2dr p2;
                        p2.x = pt2.x + dx * deltax_col + dy * deltay_col;
                        p2.y = pt2.y + dx * deltax_lig + dy * deltay_lig;
                        fenImage.push_back(img.getr(p2,NoData));
                    }
                }
                // Coeff de correlation
                double coef = correl(fenOrtho,fenImage,NoData);
                //std::cout << "Correl : "<<coef<<std::endl;
                if (coef>coefmax)
                {
                    coefmax = coef;
                    ptMaxCorrel = pt2;
                }
            }
        }
        if (verbose) std::cout << "Max de correlation : "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" correl = "<<coefmax<<std::endl;
        if (coefmax>=seuilCorrel)
            fic2 << ptSift.x<<" "<<ptSift.y<<" "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" "<<coefmax<<std::endl;
*/
    }

    //on recupere le nom de sortie sans extension
    int placePoint = -1;
    for(int l=aNameResult.size()-1;(l>=0)&&(placePoint==-1);--l)
    {
        if (aNameResult[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string aBaseNameResult = aNameResult;
    if (placePoint!=-1)
    {
        aBaseNameResult.assign(aNameResult.begin(),aNameResult.begin()+placePoint);
    }
    std::cout << "Nom de sortie : "<<aBaseNameResult<<std::endl;


     aSetDicoMesure.MesureAppuiFlottant1Im().push_back(aDicoMesure);
     MakeFileXML(aDicoAppuis,aBaseNameResult+"-S3D.xml");
     MakeFileXML(aSetDicoMesure,aBaseNameResult+"-S2D.xml");

    return EXIT_SUCCESS;
}


Pt3dr Img2Terrain(ElCamera *aCamera, TIm2D<REAL4,REAL8> *mnt, cFileOriMnt const &ori,
                  double Zinit,
                  Pt2di const &Pimg)
{
    int itMax = 10;
    double seuilZ = 1.;
    double NoData = -9999.;


    double Z = Zinit;
    Pt3dr Pterr, Pterr2;
    Pt2dr Pmnt;

    int it = 0;
    bool fin = false;
    while((it<itMax)&&!fin)
    {
        Pterr = aCamera->F2AndZtoR3(Pt2dr(Pimg.x,Pimg.y),Z);
        //std::cout << "it: "<<it<<" Pterr: "<<Pterr.x<<" "<<Pterr.y<<" "<<Pterr.z<<std::endl;
        // Position dans le MNT
        Pmnt.x = (Pterr.x-ori.OriginePlani().x)/ori.ResolutionPlani().x;
        Pmnt.y = (Pterr.y-ori.OriginePlani().y)/ori.ResolutionPlani().y;
        //std::cout << "Pmnt : "<<Pmnt<<std::endl;
        double vAlti = mnt->getr(Pmnt,NoData);
        if (vAlti == NoData)
        {
            std::cout << "Attention, le MNT ne couvre pas la zone : "<<Pmnt.x<<" "<<Pmnt.y<<std::endl;
        }
        double alti = vAlti*ori.ResolutionAlti() + ori.OrigineAlti();
        fin = (std::abs(alti - Z)<=seuilZ);
        Z = alti;
        ++it;
    }
    if (it == itMax)
        std::cout << "Attention, pas de convergence pour l'estimation des coordonnes 3D du point : "<<Pimg.x<<" "<<Pimg.y<<std::endl;

    return Pterr;
}

int Ortho(std::string const &aNameFileMNT,
          std::string const &aNameFileGrid,
          std::string const &aNameFileImage,
          double resolution,
          std::string const &aNameResult)
{
    int SzMaxImg = 4000 * 4000;

    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    std::auto_ptr<TIm2D<REAL4,REAL8> > aMntImg(createTIm2DFromFile<REAL4,REAL8>(aMntOri.NameFileMnt()));
    if (aMntImg.get()==NULL)
    {
        cerr << "Error in "<<aMntOri.NameFileMnt()<<std::endl;
        return EXIT_FAILURE;
    }

    // Chargement de la grille et de l'image
    std::auto_ptr<TIm2D<U_INT1,INT4> > img(NULL);
    Pt2di ImgSz = getImageSize(aNameFileImage);
    if (ImgSz.x * ImgSz.y < SzMaxImg)
    {
        img.reset(createTIm2DFromFile<U_INT1,INT4>(aNameFileImage));
        if (img.get()==NULL)
        {
            cerr << "Error in "<<aNameFileImage<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Chargement de la grille et de l'image
    ElAffin2D oriIntImaM2C;
    std::auto_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));

//	bool verbose=true;

    // Recherche l'emprise de l'ortho a calculer

    double NoData = 0;
    double ZMoy = 0.;
    double xmin,ymin,xmax,ymax;
    // Projection des coins de l'image pour trouver l'emprise
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(0,0));
        xmin = Pterr.x;
        ymin = Pterr.y;
        xmax = Pterr.x;
        ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(ImgSz.x,0));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(ImgSz.x,ImgSz.y));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(0,ImgSz.y));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    std::cout << "Emprise Terrain de l'image : "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<std::endl;
    Pt2dr P0Ortho(round_ni(xmin),round_ni(ymin));
    Pt2di SzOrtho((int)round_ni((xmax-xmin)/resolution),(int)round_ni((ymax-ymin)/resolution));
    std::cout << "Ortho : "<<P0Ortho.x<<" "<<P0Ortho.y<<" / "<<SzOrtho.x<<" "<<SzOrtho.y<<std::endl;

    // Creation de l'image
    TIm2D<U_INT1,INT4> anTIm2D(SzOrtho);

    // Remplissage de l'image
    for(int l=0;l<SzOrtho.y;++l)
    {
        for(int c=0;c<SzOrtho.x;++c)
        {
            Pt2di Portho(c,l);
            Pt3dr Pterr;
            Pterr.x = P0Ortho.x + c * resolution;
            Pterr.x = P0Ortho.y + l * resolution;
            // Position dans le MNT
            Pt2dr Pmnt;
            Pmnt.x = (Pterr.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
            Pmnt.y = (Pterr.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
            Pterr.z = aMntImg->getr(Pmnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
            // Position dans l'image
//			Pt2dr Pimg = aCamera->R3toF2(Pterr);
            double radio = img->getr(Pt2dr(c,l),NoData);
            anTIm2D.oset(Portho,(int)radio);
        }
    }

    // Sauvegarde
    Tiff_Im out(aNameResult.c_str(), anTIm2D.sz(),GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(anTIm2D._the_im.all_pts(),anTIm2D._the_im.in(),out.out());

    return EXIT_SUCCESS;
}


int ServiceGeoSud_TP2GCP_main(int argc, char **argv) {


    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT
    std::string aNameFileOrtho;// un fichier xml de type FileOriMnt pour l'ortho
    std::string aNameFilePOIOrtho;// un fichier dat de POI Ortho

    std::string aNameFileGrid;// un fichier GRID
    std::string aNameFileImage;// un fichier image
    std::string aNameFilePOIImage;// un fichier POI pour l'image associee a la GRID

    double seuilPixel=20;
    double seuilCorrel=0.7;
    int SzW=10;
    std::string aNameResult="result.txt";// un fichier resultat

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
                   << EAMC(aNameFileOrtho,"xml file (FileOriMnt) for the Ortho")
                   << EAMC(aNameFilePOIOrtho,"POI file for the Ortho")
                    << EAMC(aNameFileGrid,"GRID File")
                    << EAMC(aNameFileImage,"Image File")
                    << EAMC(aNameFilePOIImage,"POI file (related with the GRID and the Image)"),
     LArgMain() << EAM(seuilCorrel,"CorMin",true,"correl threshold")
                << EAM(seuilPixel,"Dmax",true,"distance threshlod (in pixel)")
                << EAM(SzW,"SzW",true,"half window size (in pixel)")
     << EAM(aNameResult,"Output","output file name")
     );

    return TP2GCP(aNameFileMNT,aNameFileOrtho,aNameFilePOIOrtho,aNameFileGrid,aNameFileImage,aNameFilePOIImage,seuilPixel,seuilCorrel,SzW,aNameResult);
}

int ServiceGeoSud_Ortho_main(int argc, char **argv) {


    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT

    std::string aNameFileGrid;// un fichier GRID
    std::string aNameFileImage;// un fichier image
    std::string aNameResult;// un fichier resultat

    double resolution;

    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
     << EAMC(aNameFileGrid,"GRID File")
     << EAMC(aNameFileImage,"Image File")
     << EAMC(resolution,"ground resolution")
     << EAMC(aNameResult,"Output file name"),
     LArgMain()
     );

    return  Ortho(aNameFileMNT,aNameFileGrid,aNameFileImage,resolution,aNameResult);
}

int ServiceGeoSud_Surf_main(int argc, char **argv){
    std::string aFullName;
    std::string aNameOut;

    int octaves=5;
    int intervals=4;
    int init_samples=2;
    int nbPoints=100;

    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(aFullName,"Input filename")
     << EAMC(aNameOut,"output filename") ,
     LArgMain()
     );

#if defined (__USE_JP2__)
    std::auto_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aFullName));
#else
    std::auto_ptr<cInterfModuleImageLoader> aRes(NULL);
#endif
    if (!aRes.get())
    {
        return 1;
    }

    Pt2di ImgSz(aRes->Sz(1).real(),aRes->Sz(1).imag());

    std::cout << "Taille de l'image  : "<<ImgSz.x<<" x "<<ImgSz.y<<std::endl;
    int tailleDalle = 4000;
    int NbX = ImgSz.x / tailleDalle;
    if (NbX*tailleDalle < ImgSz.x)
        ++NbX;
    int NbY = ImgSz.y / tailleDalle;
    if (NbY*tailleDalle < ImgSz.y)
        ++NbY;
    list<DigeoPoint> total_list;
    for(int nx = 0;nx<NbX;++nx)
    {
        for(int ny = 0;ny<NbY;++ny)
        {
            std::cout << "Traitement de la dalle : "<<nx<<" x "<<ny<<std::endl;
            int cmin = nx*tailleDalle;
            int lmin = ny*tailleDalle;
            int cmax = std::min(cmin+tailleDalle,ImgSz.x);
            int lmax = std::min(lmin+tailleDalle,ImgSz.y);
            std::cout << "Crop : "<<cmin<<" "<<lmin<<" "<<cmax<<" "<<lmax<<std::endl;

            BufferImage<unsigned short> aBuffer(cmax-cmin,lmax-lmin,1);
            std::cout << "Creation du BufferImage"<<std::endl;

            unsigned short ** ptrLine = new unsigned short * [lmax-lmin];
            for(int l=0;l<(lmax-lmin);++l)
            {
                ptrLine[l] = aBuffer.getLinePtr(l);
            }
            aRes->LoadCanalCorrel(sLowLevelIm<unsigned short>
                                  (
                                   aBuffer.getPtr(),
                                   ptrLine,
                                   std::complex<int>(cmax-cmin,lmax-lmin)
                                   ),
                                  1,//deZoom
                                  std::complex<int>(0,0),//aP0Im
                                  std::complex<int>(cmin,lmin),//aP0File
                                  std::complex<int>(cmax-cmin,lmax-lmin));
            std::cout << "Crop"<<std::endl;
            delete[] ptrLine;
            Surf s(aBuffer,octaves,intervals,init_samples,nbPoints);
            std::cout << "Nombre de points : "<<s.vPoints.size()<<std::endl;
            for(size_t i=0;i<s.vPoints.size();++i)
            {
                SurfPoint const &surfPt =s.vPoints[i];
                //std::cout << "Point : "<<surfPt.x()+cmin<<" "<<surfPt.y()+lmin<<" "<<surfPt.descripteur.size()<<std::endl;
                DigeoPoint pt;
                pt.x =surfPt.x()+cmin;
                pt.y =surfPt.y()+lmin;

                REAL8* des = new REAL8[DIGEO_DESCRIPTOR_SIZE];
                for(int d=0;d<DIGEO_DESCRIPTOR_SIZE;++d)
                {
                    if (d< (int) surfPt.descripteur.size())
                    {
                        des[d] = surfPt.descripteur[d];
                    }
                    else
                    {
                        des[d]=0.;
                    }
                }
                pt.addDescriptor(0.,des);
                delete[] des;
                total_list.push_back(pt);
            }
        }
    }

    cout << total_list.size() << " points" << endl;
    DigeoPoint::writeDigeoFile(aNameOut, total_list);


    // Verification
    {
        // Chargement des points d'interet dans l'ortho
        vector<DigeoPoint> vPts;

        DigeoPoint::readDigeoFile( aNameOut, true, vPts );
        std::cout << "Nombre de points lus: "<<vPts.size()<<std::endl;

    }

    return 0;
}

int ServiceGeoSud_GeoSud_main(int argc, char **argv){

    std::string aFullName;
    std::string aKeyGPP;
    std::string aGRIDExt("GRI");

    double ZMoy = 0.;


    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(aFullName,"Full Name (Dir+Pat)")
     << EAMC(aKeyGPP,"GPP Key"),
     LArgMain()<< EAM(aGRIDExt,"Grid",true,"GRID ext")
     );

    std::string aDir,aPat;
    SplitDirAndFile(aDir,aPat,aFullName);
    std::list<std::string> aLFile;
    cInterfChantierNameManipulateur *aICNM;
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aLFile = aICNM->StdGetListOfFile(aPat);

    std::cout << "Nombre de fichiers a traiter : "<<aLFile.size()<<std::endl;

    // On cherche l'emprise du chantier
    double xminChantier,yminChantier,xmaxChantier,ymaxChantier;
    xminChantier = yminChantier = xmaxChantier = ymaxChantier = 0;
    bool first = true;

    std::list<std::string>::const_iterator it,fin=aLFile.end();
    for(it=aLFile.begin();it!=fin;++it)
    {
        std::string aNameFileImage = (*it);
        std::cout << "fichier image : "<<aNameFileImage<<std::endl;

        // taille de l'image
        Pt2di ImgSz = getImageSize(aNameFileImage);

        // On cherche la grille correspondante
        int placePoint = -1;
        for(int l=aNameFileImage.size()-1;(l>=0)&&(placePoint==-1);--l)
        {
            if (aNameFileImage[l]=='.')
            {
                placePoint = l;
            }
        }
        std::string ext = std::string("");
        if (placePoint!=-1)
        {
            std::string baseName;
            baseName.assign(aNameFileImage.begin(),aNameFileImage.begin()+placePoint+1);
            std::string aNameFileGrid = baseName+aGRIDExt;
            std::string aNameFilePOI = baseName+"dat";
            std::cout << "fichier GRID : "<<aNameFileGrid<<std::endl;
            std::cout << "fichier POI : "<<aNameFilePOI<<std::endl;

            // Chargement de la grille et de l'image
            ElAffin2D oriIntImaM2C;
            std::auto_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));


            // On cherche l'emprise de l'image
            double xmin,ymin,xmax,ymax;
            // Projection des coins de l'image pour trouver l'emprise
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(0,0),ZMoy);
                xmin = Pterr.x;
                ymin = Pterr.y;
                xmax = Pterr.x;
                ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(ImgSz.x,0),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(ImgSz.x,ImgSz.y),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(0,ImgSz.y),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            std::cout << "Emprise Terrain de l'image : "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<std::endl;
            if (first)
            {
                first = false;
                xminChantier = xmin;
                xmaxChantier = xmax;
                yminChantier = ymin;
                ymaxChantier = ymax;
            }
            else
            {
                if (xmin<xminChantier)
                    xminChantier=xmin;
                if (xmax>xmaxChantier)
                    xmaxChantier=xmax;
                if (ymin<yminChantier)
                    yminChantier=ymin;
                if (ymax>ymaxChantier)
                    ymaxChantier=ymax;
            }

            // Extraction des POI
            {
                std::string cmdPOI="mm3d Digeo "+aNameFileImage+" -o "+aNameFilePOI;
                system(cmdPOI.c_str());
            }
        }
    }

    // On arrondi
    xminChantier = (int)(xminChantier-1);
    xmaxChantier = (int)(xmaxChantier+1);
    yminChantier = (int)(yminChantier-1);
    ymaxChantier = (int)(ymaxChantier+1);

    double resolution = 10.;//10m
    int NC = (xmaxChantier-xminChantier)/resolution;
    int NL = (ymaxChantier-yminChantier)/resolution;

    std::cout << std::fixed << "Emprise du chantier : "<<xminChantier<<" "<<yminChantier<<" "<<xmaxChantier<<" "<<ymaxChantier<<std::endl;

    // Extraction de l'ortho
    {
        std::ostringstream oss;
        oss << std::fixed << "curl -o ortho.tif -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminChantier<<","<<yminChantier<<","<<xmaxChantier<<","<<ymaxChantier<<"&CRS=EPSG:2154&WIDTH="<<NC<<"&HEIGHT="<<NL<<"\"";
        std::cout << "commande : "<<oss.str()<<std::endl;
        system(oss.str().c_str());
    }
    // Extraction des POI
    {
        std::string cmdPOI="mm3d Digeo ortho.tif -o ortho.dat";
        system(cmdPOI.c_str());
    }
    // Extraction du MNT
    {
        std::ostringstream oss;
        oss << std::fixed << "curl -o mnt.bil -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/x-bil;bits=32&BBOX="<< xminChantier<<","<<yminChantier<<","<<xmaxChantier<<","<<ymaxChantier<<"&CRS=EPSG:2154&WIDTH="<<NC<<"&HEIGHT="<<NL<<"\"";
        std::cout << "commande : "<<oss.str()<<std::endl;
        system(oss.str().c_str());


        //echo 'NROWS 675\nNCOLS 1769\nNBANDS 1\nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > mnt.HDR
        std::ostringstream ossHdr;
        ossHdr << "echo 'NROWS "<<NL<<"\nNCOLS "<<NC<<"\nNBANDS 1\"nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > mnt.HDR";
        system(ossHdr.str().c_str());
    }

    /*
    // Il faut convertir cette emprise en coordonnees geographique
    //    command = "cs2cs "+targetSyst+" +to +proj=latlon +datum=WGS84 +ellps=WGS84 -f %.12f -s  processing/indirect_ptCarto.txt >  processing/indirect_ptGeo.txt";
    //echo 368225.296078 6555181.166326 | cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84
    //echo 368225.296078 6555181.166326 > tempCarto.txt
    //cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84 -f %.12f tempCarto.txt > tempGeo.txt
    {
        std::ofstream ficTempCarto("tempCarto.txt");
        ficTempCarto << std::fixed << xminChantier<<" "<<yminChantier<<std::endl;
        ficTempCarto << std::fixed << xminChantier<<" "<<ymaxChantier<<std::endl;
        ficTempCarto << std::fixed << xmaxChantier<<" "<<ymaxChantier<<std::endl;
        ficTempCarto << std::fixed << xmaxChantier<<" "<<yminChantier<<std::endl;
    }
    std::string proj4cmd("cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84 -f %.12f tempCarto.txt > tempGeo.txt");
    command(proj4cmd.c_str());
    double latmax,lonmin;
    {
        bool first = true;
        std::iftream ficTempGeo("tempGeo.txt");
        double lon,lat;
        ficTempGeo >> lon >> lat;
        if (fic.good())
        {
            if (first)
            {
                first = false;
                latmax = lat;
                lonmin = lon;
            }
            else
            {
                if (lat>latmax)
                    latmax=lat;
                if (lon<lonmin)
                    lonmin=lon;
            }
        }
    }
    std::cout << "Emprise Geo : "<<latmax<<" "<<lonmin<<std::endl;
    */

    // On extrait les POI de l'ortho

    // Export des fichiers
    // PtAppuis.txt (NumPt,lon,lat,alti)
    // Appuis.txt (NumPt,NumImage,ligne,colonne)
    // PtLiaisons.txt (NumPt,alti)
    // Liaisons.txt (NumPt,NumImage,ligne,colonne)


    return 0;
}

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic pop
    #endif
#endif


