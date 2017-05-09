#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic push
    #endif
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

// to do : gerer cas central...

///
///
///
double correlVec(std::vector<double> const &f1,std::vector<double> const &f2)
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
    double pdtVar = (sa2-pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar!=0.)
        return (sab-sa*sb/N)/sqrt(pdtVar);
    return 0.;
}

///
///
///
std::string getGPPAccess(const std::string aHttpProxy)
{
    std::ostringstream oss;
    oss << g_externalToolHandler.get( "curl" ).callName() << " -H='Referer: http://localhost' ";
    if (!aHttpProxy.empty())
        oss << "-x "<<aHttpProxy;
    return oss.str();
}

///
///
///
void getVignetteGPP(const std::string nomDalleOrtho, const std::string aKeyGPP, const std::string aHttpProxy,
                    const Pt3dr& ptSol, const size_t sizeCropBDOrtho,
                    const float resolBDOrtho, const float resolImageSat)
{
    std::string gppAccess = getGPPAccess(aHttpProxy);
    
    //dalle
    double xminDalle = ptSol.x - sizeCropBDOrtho*resolBDOrtho;
    double yminDalle = ptSol.y - sizeCropBDOrtho*resolBDOrtho;
    double xmaxDalle = ptSol.x + sizeCropBDOrtho*resolBDOrtho;
    double ymaxDalle = ptSol.y + sizeCropBDOrtho*resolBDOrtho;
    
    //taille de la dalle
    int ncDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
    int nlDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
    
    //recuperation de l'ortho
    std::ostringstream ossGPP;
    ossGPP << std::fixed << gppAccess << " -o "<<nomDalleOrtho<< " \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";
    
    system(ossGPP.str().c_str());
}

///
///
///
void getVecFromTIm2D(const std_unique_ptr<TIm2D<U_INT2,INT4> >& cropImg, std::vector<double>& fenImage,
                     const double NoData)
{
    for (size_t l(0); l<cropImg->sz().y; l++)
    {
        for (size_t c(0); c<cropImg->sz().x; c++)
        {
            fenImage.push_back(cropImg->getr(Pt2dr(c,l),NoData));
        }
    }
}

///
///
///
double getCorelCoef(const Pt2di& ptImage, const Pt2di& orthoSz, const std::string nomImage,
                    const std::vector<double> fenRef, const double NoData)
{
    std::vector<double> fenImage_i;
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg (createTIm2DFromFile<U_INT2,INT4>(nomImage,ptImage,orthoSz));
    getVecFromTIm2D(cropImg,fenImage_i,NoData);
    
    //correlation :
    return correlVec(fenRef,fenImage_i);
}

///
///
///
double minVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]<ret)
            ret = c[i];
    return ret;
}
double maxVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]>ret)
            ret = c[i];
    return ret;
}
///
///
///
std_unique_ptr<TIm2D<U_INT2,INT4> > calcPseudoOrtho(const std::string& aNameImage, const Pt3dr& ptCentre, const Pt2di& size,
                                                    const float resolOrtho, std_unique_ptr<ElCamera>& aCam, int buffer)
{
    std_unique_ptr<TIm2D<U_INT2,INT4> > vignette (new TIm2D<U_INT2,INT4>(size));
    
    Pt3dr PtNO (ptCentre.x-size.x*resolOrtho,ptCentre.y+size.y*resolOrtho,ptCentre.z); Pt2dr ptINO = aCam->Ter2Capteur(PtNO);
    Pt3dr PtNE (ptCentre.x+size.x*resolOrtho,ptCentre.y+size.y*resolOrtho,ptCentre.z); Pt2dr ptINE = aCam->Ter2Capteur(PtNE);
    Pt3dr PtSO (ptCentre.x-size.x*resolOrtho,ptCentre.y-size.y*resolOrtho,ptCentre.z); Pt2dr ptISO = aCam->Ter2Capteur(PtSO);
    Pt3dr PtSE (ptCentre.x+size.x*resolOrtho,ptCentre.y-size.y*resolOrtho,ptCentre.z); Pt2dr ptISE = aCam->Ter2Capteur(PtSE);
    
    std::vector<double> vc; vc.push_back(ptINO.x); vc.push_back(ptINE.x); vc.push_back(ptISO.x); vc.push_back(ptISE.x);
    std::vector<double> vl; vl.push_back(ptINO.y); vl.push_back(ptINE.y); vl.push_back(ptISO.y); vl.push_back(ptISE.y);

    double cmin = minVec(vc);
    double cmax = maxVec(vc);
    double lmin = minVec(vl);
    double lmax = maxVec(vl);
    
    Pt2di minIm (cmin-buffer,lmin-buffer);
    Pt2di sizeIm (cmax-cmin+2,lmax-lmin+2);
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg_cl (createTIm2DFromFile<U_INT2,INT4>(aNameImage,minIm,sizeIm));
    
    for (size_t l(0); l<size.y ; l++)
    {
        for (size_t c(0); c<size.x ; c++)
        {
            Pt3dr pt (PtNO.x + c*resolOrtho, PtNO.y - l*resolOrtho, PtNO.z);
            Pt2dr ptImage = aCam->Ter2Capteur(pt);

            Pt2dr ptImageCrop (ptImage.x-cmin+buffer,ptImage.y-lmin+buffer);
            double radio = cropImg_cl->getr(ptImageCrop);
            vignette->oset(Pt2di(c,l),(int)radio);
        }
    }
    
    return vignette;
}

///
///
///
Pt2di ImageSize(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
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
    
#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get())
        {
            return Std2Elise(aRes->Sz(1));
        }
    }
#endif
    
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    return aTif.sz();
}

///
///
///
bool getPoints(const std::string aBDCFileName, std::map<size_t,Pt3dr>& mGCP)
{
    std::ifstream fic (aBDCFileName.c_str());
    if (!fic.good())
    {
        cerr << "Error in reading "<<aBDCFileName<< " !" << std::endl;
        return false;
    }
    
    std::string line;
    while ( std::getline(fic, line ) )
    {
        Pt3dr Pt; size_t nPt(0);
        std::istringstream iss4(line);
        iss4 >> nPt >> Pt.x >> Pt.y >> Pt.z;
        mGCP.insert(std::pair<size_t,Pt3dr>(nPt,Pt));
    }
    
    return true;
}

///
///
///
int GCP_From_BDCarrefour_main(int argc, char **argv)
{
    bool verbose  (1);
    bool verbose2 (1);
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Get BDOrtho control point for bundle" << std::endl;
    
    std::string aBDCFileName;
    std::string aOutDir;
    std::string aNameImage,aNameImageOri;
    
    //pour connexion GPP :
    std::string aKeyGPP;
    std::string aHttpProxy;
    
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aBDCFileName,"Name of the GCP file")
                    << EAMC(aNameImage,"Name of the image")
                    << EAMC(aNameImageOri,"Name of the image orientation")
                    << EAMC(aOutDir,"Name of output directory")
                    << EAMC(aKeyGPP,"GPP Key"),
         LArgMain() << EAM(aHttpProxy,"Proxy",true,"http proxy for GPP access")
    );
    
    //lecture des points de contrôle
    std::map<size_t,Pt3dr> mGCP;
    if (!getPoints(aBDCFileName,mGCP)) return EXIT_FAILURE;
    
   if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Nb of points : " << mGCP.size() << std::endl;

    // chargement de l'orientation de l'image sat
    ElAffin2D oriIntImaM2C;
    Pt2di ImgSz = ImageSize(aNameImage);
    std_unique_ptr<ElCamera> aCam (new cCameraModuleOrientation(new OrientationGrille(aNameImageOri),ImgSz,oriIntImaM2C));
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] input image resolution : "<< aCam->ResolutionSol() << std::endl;

    float resolImageSat=aCam->ResolutionSol();//todo : le sol doit etre a 0. : a voir si ce n'est pas le cas.
    float resolBDOrtho = 0.20;//connu a priori
    
    size_t sizeCropBDOrtho = 50;
    double NoData = -1.;
    int marge = 10;
    double coefOk=0.9;
    int buffer (1);
    
    //  lecture de la BDOrtho et recuperation d'un crop
    //  on trouve le point par correlation
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] for each points..." << std::endl;
    std::map<size_t,Pt3dr>::iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
    std::map<size_t,Pt2dr> mPix; std::map<size_t,Pt2dr>::iterator itPix;
    for (;itGCP!=endGCP;itGCP++)
    {
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << itGCP->first << std::endl;
        
        //recuperation de la vignette
        std::ostringstream ossNomOrtho;
        ossNomOrtho << "Ortho_"<<itGCP->first<<".tif";
        std::string nomDalleOrtho = ossNomOrtho.str();
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << itGCP->first << " : vignette" << std::endl;
        getVignetteGPP(nomDalleOrtho,aKeyGPP,aHttpProxy,itGCP->second,sizeCropBDOrtho,resolBDOrtho,resolImageSat);
        
        //info de l'image :
        int ncDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
        int nlDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
        Pt2di Sz (ncDalle,nlDalle);
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << Sz.x << " " << Sz.y << std::endl;
        
        //BDOrtho => tab
        std::vector<double> fenOrtho;
        std_unique_ptr<TIm2D<U_INT2,INT4> > cropBDOrtho (createTIm2DFromFile<U_INT2,INT4>(nomDalleOrtho,Pt2di(0,0),Sz));
        getVecFromTIm2D(cropBDOrtho,fenOrtho,NoData);
        
        //point dans l'image :
        Pt2dr ptImage = aCam->Ter2Capteur(itGCP->second);
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point in input image " << ptImage.x << " " << ptImage.y << std::endl;
        double CoefCorelMax(-1);

        //calcul d'une pseudoOrtho :
        Pt2di orthoSz (Sz.x+2*marge,Sz.y+2*marge);
        std_unique_ptr<TIm2D<U_INT2,INT4> > pseudoOrtho = calcPseudoOrtho(aNameImage,itGCP->second,orthoSz,resolBDOrtho,aCam,buffer);
        
        //export de l'image d'ortho
        std::ostringstream ossNomPseudoOrtho;
        ossNomPseudoOrtho << "PseudoOrtho_"<<itGCP->first<<".tif";
        std::string nomPseudoOrtho = ossNomPseudoOrtho.str();
        Tiff_Im out(nomPseudoOrtho.c_str(), pseudoOrtho->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
        ELISE_COPY(pseudoOrtho->_the_im.all_pts(),pseudoOrtho->_the_im.in(),out.out());
        
        //on regarde si le pt image est bon :
        // si la corel marche bien sur un pixel autour du point de loc, dans le doute on garde la loc...
        // todo : faire mieux...
        
        //correlations :
        for (size_t l(0); l<orthoSz.y-Sz.y;l++)
        {
            for (size_t c(0); c<orthoSz.x-Sz.x;c++)
            {
                double cofCorel =  getCorelCoef(Pt2di(c,l),Sz,nomPseudoOrtho,fenOrtho,NoData);

                if(cofCorel>CoefCorelMax)
                {
                    Pt2di ptCentralPO (c+Sz.x/2,l+Sz.y/2);
                    Pt3dr ptCentralPOTer ( itGCP->second.x - resolBDOrtho*(Sz.x/2 - ptCentralPO.x),
                                           itGCP->second.y - resolBDOrtho*(Sz.y/2 - ptCentralPO.y),
                                           itGCP->second.z);
                    Pt2dr ptImage_i = aCam->Ter2Capteur(ptCentralPOTer);
                    
                    CoefCorelMax = cofCorel;
                    if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] " << itGCP->first << " : " << cofCorel << "  pt (" << ptImage_i.x <<"," <<ptImage_i.y<<")" << std::endl;
                    
                    itPix = mPix.find(itGCP->first);
                    Pt2dr ptr (ptImage_i.x,ptImage_i.y);
                    if (itPix==mPix.end())
                        mPix.insert(std::pair<size_t,Pt2dr>(itGCP->first,ptr));
                    else
                        itPix->second = ptr;
                }
            }
        }
        
        //sortie de la meilleure image :
        {
            std::ostringstream ossNomVignetteResult;
            ossNomVignetteResult << "Vignette_"<<itGCP->first<<".tif";
            std::string nomVignetteResult = ossNomVignetteResult.str();

            itPix = mPix.find(itGCP->first);
            if (itPix == mPix.end()) return EXIT_FAILURE;
            
            if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] (" << ptImage.x << ";" << ptImage.y << ") => (" << itPix->second.x << ";" << itPix->second.y << ")" << std::endl;
            
            int cmin = itPix->second.x - Sz.x/2;
            int lmin = itPix->second.y - Sz.y/2 ;
            
            std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg (createTIm2DFromFile<U_INT2,INT4>(aNameImage,Pt2di(cmin,lmin),Sz));
            Tiff_Im out(nomVignetteResult.c_str(), cropImg->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
            ELISE_COPY(cropImg->_the_im.all_pts(),cropImg->_the_im.in(),out.out());
        }
        
    }
    
    return 0;
}

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic pop
    #endif
#endif


