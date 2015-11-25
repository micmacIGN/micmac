#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"

/** Development based on
 CARTOSAT-1 DEM EXTRACTION CAPABILITY STUDY OVER SALON AREA
 R. Gachet & P. Fave
**/



class AffCamera
{
 public:
    AffCamera(string aFilename, int index, bool activate=false): filename (StdPrefix(aFilename)), mIndex(index), activated(activate)
    {
        // Loading the GRID file
        ElAffin2D oriIntImaM2C;
        Pt2di Sz(10000,10000);
        mCamera =  new cCameraModuleOrientation(new OrientationGrille(aFilename),Sz,oriIntImaM2C);

        //Affinity parameters
        vP.push_back(0);
        vP.push_back(1);
        vP.push_back(0);
        vP.push_back(0);
        vP.push_back(0);
        vP.push_back(1);
    }

    ~AffCamera()
    {
        if (mCamera)
            delete mCamera;
    }

    ///
    /// \brief update affinity parameters
    /// \param sol unknowns matrix
    ///
    void updateParams(ElMatrix <double> const &sol)
    {
        cout << "Init solution of cam " << filename <<endl;
        for (size_t aK=0; aK< vP.size(); aK++)
            cout << vP[aK] <<" ";
        cout << endl;

        for (size_t aK=0; aK< vP.size(); aK++)
            vP[aK] += sol(0,(int)aK);

        cout << "Updated solution: "<<endl;
        for (size_t aK=0; aK< vP.size(); aK++)
            cout << vP[aK] <<" ";
        cout << endl;
    }

    ElCamera* Camera() { return mCamera; }

    // the 6 parameters of affinity
    // colc = vP[0] + vP[1] * col + vP[2] * lig
    // rowc = vP[3] + vP[4] * col + vP[5] * lig
    vector <double> vP;

    Pt2dr apply(Pt2dr const &ptImg2)
    {
        return Pt2dr(vP[0] + vP[1] * ptImg2.x + vP[2] * ptImg2.y,
                     vP[3] + vP[4] * ptImg2.x + vP[5] * ptImg2.y);
    }

    void activate(bool aVal) { activated = aVal; }
    bool isActivated() { return activated; }

    string name() { return filename; }

    int index() { return mIndex; }

protected:

    ///
    /// \brief image filename
    ///
    string filename;

    int mIndex;

    ElCamera* mCamera;

    bool activated; //should this camera be estimated

};

class ImageMeasure
{
public:
    ImageMeasure(Pt2dr aPt, AffCamera* aCam):_ptImg(aPt),_idx(aCam->index()), _imgName(aCam->name()){}

    Pt2dr  pt()       { return _ptImg; }
    string imgName()  { return _imgName; }
    int    imgIndex() { return _idx;   }

private:
    Pt2dr _ptImg;       // image coordinates (in pixel)
    int   _idx;         // index of AffCamera

    string _imgName;    // name of AffCamera

    //bool valid; // should this measure be used for estimation
};

class Observation
{
public:
    Observation(map <int, AffCamera *> *aMap, string aName = "", bool aValid = false, Pt3dr aPt = Pt3dr(0.f,0.f,0.f)):
        pMapCam(aMap),
        valid(aValid),
        _PtTer(aPt),
        _ptName(aName){}

    virtual Pt3dr getCoord()=0;

    string ptName() { return _ptName; }

    void addImageMeasure(ImageMeasure const &aMes) { vImgMeasure.push_back(aMes); }

    Pt2dr computeImageDifference(int index,
                                   Pt3dr pt,
                                   double aA0,
                                   double aA1,
                                   double aA2,
                                   double aB0,
                                   double aB1,
                                   double aB2);

    Pt2dr computeImageDifference(int index, Pt3dr pt);

    map <int, AffCamera*> *pMapCam;

    vector <ImageMeasure> vImgMeasure;

    bool valid; // should this observation be used for estimation?

protected:
    Pt3dr _PtTer;

    string _ptName;
};

Pt2dr Observation::computeImageDifference(int index,
                                          Pt3dr pt,
                                          double aA0, double aA1, double aA2, double aB0, double aB1, double aB2)
{
    ImageMeasure* aMes = &vImgMeasure[index];

    Pt2dr ptImg = aMes->pt();

    map<int, AffCamera *>::const_iterator iter = pMapCam->find(aMes->imgIndex());
    AffCamera* cam = iter->second;

    Pt2dr ptImgC(aA0 + aA1 * ptImg.x + aA2 * ptImg.y,
                 aB0 + aB1 * ptImg.x + aB2 * ptImg.y);

    Pt2dr proj = cam->Camera()->R3toF2(pt);

    return ptImgC - proj;
}

Pt2dr Observation::computeImageDifference(int index, Pt3dr pt)
{
    ImageMeasure* aMes = &vImgMeasure[index];

    Pt2dr ptImg = aMes->pt();

    map<int, AffCamera *>::const_iterator iter = pMapCam->find(aMes->imgIndex());
    AffCamera* cam = iter->second;

    Pt2dr ptImgC = cam->apply(ptImg);

    Pt2dr proj = cam->Camera()->R3toF2(pt);

    return ptImgC - proj;
}

class GCP : public Observation
{
public:
    GCP(map <int, AffCamera*> *aMap, string aPtName="", bool valid = true, Pt3dr aPt=Pt3dr(0.f,0.f,0.f)):Observation(aMap,aPtName,valid, aPt){}

    Pt3dr getCoord() { return _PtTer; }
};

class TiePoint : public Observation
{
public:
    TiePoint(map <int, AffCamera *> *aMap, string aPtName="", bool valid = true): Observation(aMap, aPtName, valid){}

    Pt3dr getCoord();
};

Pt3dr TiePoint::getCoord()
{
    if (vImgMeasure.size() == 2)
    {
        map<int, AffCamera *>::iterator iter1 = pMapCam->find(vImgMeasure[0].imgIndex());
        map<int, AffCamera *>::iterator iter2 = pMapCam->find(vImgMeasure[1].imgIndex());

        AffCamera* cam1 = iter1->second;
        AffCamera* cam2 = iter2->second;

        Pt2dr P1 = cam1->apply(vImgMeasure[0].pt());
        Pt2dr P2 = cam2->apply(vImgMeasure[1].pt());

        return cam1->Camera()->PseudoInter(P1,*cam2->Camera(),P2);
    }
    else
    {
        vector<ElSeg3D>  aVS;
        for ( size_t aK=0; aK < vImgMeasure.size(); aK++ )
        {
            map<int, AffCamera *>::iterator iter = pMapCam->find(vImgMeasure[aK].imgIndex());

            if (iter != pMapCam->end())
            {
                AffCamera* aCam = iter->second;

                Pt2dr aPN = aCam->apply(vImgMeasure[aK].pt());

                aVS.push_back(aCam->Camera()->F2toRayonR3(aPN));
            }
        }

        return ElSeg3D::L2InterFaisceaux(0,aVS);
    }
}

double compute2DGroundDifference(Pt2dr const &ptImg1,
                                 AffCamera* Cam1,
                                 Pt2dr const &ptImg2,
                                 AffCamera* Cam2)
{
    double z = Cam1->Camera()->PseudoInter(ptImg1,*Cam2->Camera(),ptImg2).z;

    Pt3dr ptTer1 = Cam1->Camera()->ImEtProf2Terrain(ptImg1,z);
    Pt2dr ptImg2C(Cam2->apply(ptImg2));
    Pt3dr ptTer2 = Cam2->Camera()->ImEtProf2Terrain(ptImg2C,z);

    return square_euclid(Pt2dr(ptTer1.x - ptTer2.x,ptTer1.y - ptTer2.y));
}

//! Abstract class for shared methods
class RefineModelAbs
{
protected:
    map <int, AffCamera*> mapCameras;

    vector <Observation*> vObs;

    ///
    /// \brief normal matrix for least squares estimation
    ///
    ElMatrix<double> _N;
    ///
    /// \brief matrix for least squares estimation
    ///
    ElMatrix<double> _Y;

    size_t numUnk;

    bool _verbose;

    int iteration;

public:

    vector <Observation*> getObs() { return vObs; }

    ///
    /// \brief constructor (loads GRID files, tie-points and filter tie-points on 2D ground difference)
    /// \param aNameFileGridMaster Grid file for master image
    /// \param aNameFileGridSlave Grid file for slave image
    /// \param aNamefileTiePoints Tie-points file
    ///
    RefineModelAbs(string const &aFullDir, string imgsExtension = ".tif", bool filter = false):_N(1,1,0.),_Y(1,1,0.),numUnk(6), _verbose(false), iteration(0)
    {
        string aDir, aPat;
        SplitDirAndFile(aDir,aPat,aFullDir);

        list<string> aVFiles = RegexListFileMatch(aDir, aPat, 1, false);
        list<string>::iterator itr = aVFiles.begin();
        for(;itr != aVFiles.end(); itr++)
        {
            string aNameFileGrid1 = *itr;

            list<string>::iterator it = itr; it++;
            for(;it != aVFiles.end(); it++)
            {
                string aNameFileGrid2 = *it;

                // Loading the GRID file
                AffCamera* Cam1 = findCamera(aDir + aNameFileGrid1);
                AffCamera* Cam2 = findCamera(aDir + aNameFileGrid2);

                // Loading the Tie Points
                cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

                string aNameFileTiePoints =  aDir +  aICNM->Assoc1To2
                                         (
                                            "NKS-Assoc-CplIm2Hom@@dat",
                                            StdPrefixGen(aNameFileGrid1) + imgsExtension,
                                            StdPrefixGen(aNameFileGrid2) + imgsExtension,
                                            true
                                         );

                if ( !ELISE_fp::exist_file(aNameFileTiePoints) ) aNameFileTiePoints = StdPrefixGen(aNameFileTiePoints) + ".txt";

                if ( !ELISE_fp::exist_file(aNameFileTiePoints) ) cout << "file missing: " << StdPrefixGen(aNameFileTiePoints) + ".dat" << endl;
                else
                {
                    cout << "reading: " << aNameFileTiePoints << endl;

                    ElPackHomologue aPackHomol = ElPackHomologue::FromFile(aNameFileTiePoints);
                    //cout << "taille " << aPackHomol.size() << endl;

                    if (aPackHomol.size() == 0)
                    {
                        cout << "Error in RefineModelAbs: no tie-points" << endl;
                        return;
                    }

                    int rPts_nb = 0; //rejected points number
                    int mulTP_nb =0; //multiple points number

                    ElPackHomologue::const_iterator iter = aPackHomol.begin();
                    for (;iter!=aPackHomol.end();++iter )
                    {
                        ElCplePtsHomologues aCple = iter->ToCple();

                        Pt2dr P1 = aCple.P1();
                        Pt2dr P2 = aCple.P2();

                        //cout << "P1 = "<<P1.x<<" " <<P1.y << endl;
                        //cout << "P2 = "<<P2.x<<" " <<P2.y << endl;

                        if ((filter) && (compute2DGroundDifference(P1, Cam1, P2, Cam2) > 100.))
                        {
                            rPts_nb++;
                            //cout << "Couple with 2D ground difference > 10 rejected" << endl;
                        }
                        else
                        {
                            ImageMeasure imMes1(P1,Cam1);
							ImageMeasure imMes2(P2, Cam2);

							bool found = false;

                            if (mapCameras.size() > 2)
							{
                                //algo brute force (à améliorer)
                                for (size_t aK=0; aK < nObs(); ++aK)
                                {
                                    TiePoint* TP = dynamic_cast<TiePoint*> (vObs[aK]);
                                    for (size_t bK=0; bK < TP->vImgMeasure.size(); bK++)
                                    {
                                        ImageMeasure *imMes3 = &TP->vImgMeasure[bK];
                                        if ((imMes3->imgName() != Cam1->name()) && (imMes3->pt() == P1))
                                        {
                                            TP->addImageMeasure(imMes1);
                                            mulTP_nb++;
                                            // cout << "multiple point: " << P1.x << " " << P1.y << " found in " << imMes3->idx << " and " << imMes1.idx << endl;
                                            found = true;
                                        }
                                        else if ((imMes3->imgName() != Cam2->name()) && (imMes3->pt() == P2))
                                        {
                                            TP->addImageMeasure(imMes2);
                                            // cout << "multiple point: " << P2.x << " " << P2.y << " found in " << imMes3->idx << " and " << imMes2.idx << endl;
                                            mulTP_nb++;
                                            found = true;
                                        }
                                    }
                                }
                            }

                            if (!found)
                            {
                                TiePoint *TP = new TiePoint(&mapCameras);

                                TP->addImageMeasure(imMes1);
                                TP->addImageMeasure(imMes2);

                                vObs.push_back(TP);

                                //cout << "vObs size : " << nObs() << endl;
                            }
                        }
                    }

                    cout << "Number of tie points: "  << aPackHomol.size() << endl;
                    cout << "Number of multiple tie points: " << mulTP_nb << endl;

                    if (filter)
                    {
                        cout << "Number of rejected tie points: " << rPts_nb << endl;
                        cout << "Final number of tie points: "    << aPackHomol.size() - rPts_nb << endl;
                    }
                }
            }
        }

        cout << "mapCameras.size= " << mapCameras.size() << endl;
    }


    ///
    /// \brief image distance sum for all tie points (to compute RMS)
    /// \return sum of residuals (square distance - to avoid using sqrt (faster) )
    ///
    double sumRes(int &nbMes)
    {
        nbMes = 0;
        double sumRes = 0.;

        //pour chaque point de liaison
        for (size_t aK=0; aK < nObs();++aK)
        {
            Observation* TP = vObs[aK];
            Pt3dr PT = TP->getCoord();

            //pour chaque image ou le point de liaison est vu
            for(size_t i=0;i<TP->vImgMeasure.size();++i, ++nbMes)
            {
                Pt2dr D = TP->computeImageDifference((int)i, PT);
                sumRes += square_euclid(D);
            }
        }
        return sumRes;
    }

    ///
    /// \brief debug matrix
    /// \param mat matrix to write
    ///
    void printMatrix(ElMatrix <double> const & mat, string name="")
    {
        cout << "-------------------------"<<endl;
        cout << "Matrix " << name << " : " << endl;
        for(int j=0;j<mat.Sz().y;++j)
        {
            for(int i=0;i<mat.Sz().x;++i)
                cout << mat(i,j) << " ";

            cout << endl;
        }
        cout << "-------------------------"<<endl;
    }

    ///
    /// \brief check if a new iteration should be run and write result file (at the step before exiting loop)
    /// \param iniRMS rms before system solve
    /// \param numObs system number of observations
    /// \return
    ///
    bool launchNewIter(double iniRMS, int numObs)
    {
        double res = sumRes(numObs);
        cout << "res= " << res << " numObs= " << numObs << endl;

        if (numObs)
        {
            cout << "RMS_init = " << iniRMS << endl;

            double curRMS = sqrt(res/numObs);

            if (curRMS>=iniRMS)
            {
                cout << "curRMS = "<<curRMS<<" / iniRMS = "<<iniRMS<<endl;
                cout << "No improve: end at iteration "<< iteration <<endl;

				//ecriture dans un fichier des coefficients en vue d'affiner la grille
				//consommateur en temps => todo: stocker les parametres de l'iteration n-1
				map<int, AffCamera *>::const_iterator iter = mapCameras.begin();
				for (size_t aK = 0; iter != mapCameras.end(); ++iter, ++aK)
				{
					AffCamera* cam = iter->second;
					string name = StdPostfix(cam->name());
					ofstream fic(("refine/" + name + "_refineCoef.txt").c_str());
					cout << "Writing refineCoef file : refine" + name + "_refineCoef.txt" << endl;
					fic << setprecision(15);
					fic << cam->vP[0] << " " << cam->vP[1] << " " << cam->vP[2] << " " << cam->vP[3] << " " << cam->vP[4] << " " << cam->vP[5] << " " << endl;
				}
				return false;
			}
            cout << "RMS_after = " << curRMS << endl;
            iteration++;
            return true;
        }
        else
        {
            cout << "Error in launchNewIter numObs=0" << endl;
            return false;
        }
    }

    ///
    /// \brief estimates affinity parameters
    ///
    virtual void solve()=0;

    ///
    /// \brief computes the observation matrix for one iteration
    /// \return boolean stating if system is solved (need new iteration)
    ///
    virtual bool computeObservationMatrix()=0;

    virtual ~RefineModelAbs(){}

    AffCamera *findCamera(string aFilename)
    {
        map<int, AffCamera *>::const_iterator it = mapCameras.begin();
        for (;it!=mapCameras.end();++it)
        {
            if (it->second->name() == StdPrefixGen(aFilename)) return it->second;
        }
        AffCamera* Cam = new AffCamera(aFilename, (int)mapCameras.size(), mapCameras.size()!=0);
        mapCameras.insert(pair <int,AffCamera*>((int)mapCameras.size(), Cam));
        return Cam;
    }

    size_t nObs() { return vObs.size(); }
};


//!
class RefineModelGlobal: public RefineModelAbs
{

public:
    RefineModelGlobal(string const &aPattern,
                      string const &aImgsExtension,
                      string const &aNameFileGCP = "",
                      string const &aNameFilePointeIm = "",
                      string const &aNameFileMNT = "",
                      bool filter = false):
        RefineModelAbs(aPattern, aImgsExtension, filter)
    {
        if ((aNameFileMNT != "") && (ELISE_fp::exist_file(aNameFileMNT)))
        {
            // Chargement du MNT
            _MntOri = StdGetFromPCP(aNameFileMNT,FileOriMnt);
            if (_verbose) cout << "DTM size : "<<_MntOri.NombrePixels().x<<" "<<_MntOri.NombrePixels().y<<endl;
            std_unique_ptr<TIm2D<REAL4,REAL8> > Img(createTIm2DFromFile<REAL4,REAL8>(_MntOri.NameFileMnt()));

			#if __cplusplus > 199711L | _MSC_VER == 1800
            _MntImg = std::move(Img);
            #else
            _MntImg = Img;
            #endif
            if (_MntImg.get()==NULL) cerr << "Error in "<< _MntOri.NameFileMnt() <<endl;
            //TODO: utiliser le MNT comme contrainte en Z
        }

        if ((aNameFileGCP != "") && (ELISE_fp::exist_file(aNameFileGCP)))
        {
            cDicoAppuisFlottant aDico = StdGetFromPCP(aNameFileGCP,DicoAppuisFlottant);
            if (_verbose)
                cout << "Nb GCP " <<  aDico.OneAppuisDAF().size() << endl;
            list<cOneAppuisDAF> & aLGCP =  aDico.OneAppuisDAF();

            for (
                     list<cOneAppuisDAF>::iterator iT = aLGCP.begin();
                     iT != aLGCP.end();
                     iT++
                )
            {
                if (_verbose) cout << "Point : " << iT->NamePt() << " " << iT->Pt() << endl;

                vObs.push_back(new GCP(&mapCameras, iT->NamePt(), true, iT->Pt()));
            }
        }

        if ((aNameFilePointeIm != "") && (ELISE_fp::exist_file(aNameFilePointeIm)))
        {
            cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aNameFilePointeIm,SetOfMesureAppuisFlottants);
            if (_verbose)
                cout << "Nb GCP img " << aDico.MesureAppuiFlottant1Im().size() << endl;
            list<cMesureAppuiFlottant1Im> & aLGCP =  aDico.MesureAppuiFlottant1Im();

            for (
                     list<cMesureAppuiFlottant1Im>::iterator iT = aLGCP.begin();
                     iT != aLGCP.end();
                     iT++
                )
            {
                if (_verbose)
                    cout << "Image : " << StdPrefixGen(iT->NameIm()) << endl;

                list< cOneMesureAF1I > & aLPIm = iT->OneMesureAF1I();

                for (
                     list<cOneMesureAF1I>::iterator iTP = aLPIm.begin();
                     iTP != aLPIm.end();
                     iTP++
                    )
                {
                    if (_verbose)
                        cout << "Point : " << iTP->NamePt() << " " << iTP->PtIm() << endl;

                    AffCamera* cam = NULL;
                    for (
                         map<int, AffCamera *>::iterator iter = mapCameras.begin();
                         iter != mapCameras.end();
                         iter++
                        )
                    {
                        if (NameWithoutDir(iter->second->name()) == StdPrefixGen(iT->NameIm()))
                        {
                            cam = iter->second;
                            break;
                        }
                    }

                    if (cam)
                    {
                        ImageMeasure im( iTP->PtIm(), cam );

                        for (unsigned int aK=0; aK<vObs.size();++aK)
                        {
                            if (vObs[aK]->ptName() == iTP->NamePt() )
                            {
                                vObs[aK]->addImageMeasure(im);
                                if (_verbose)
                                    cout << "add img measure to GCP " << vObs[aK]->ptName() << endl;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    void addObs(int pos, const ElMatrix<double> &obs, const ElMatrix<double> &ccc, const double p, const double res)
    {
        bool verbose = _verbose;

        double pdt = 1./(p*p);

        ElMatrix <double> C  = ccc.transpose()*ccc*pdt;  //1 - ajouter en 0,0
        ElMatrix <double> C1 = ccc.transpose()*obs*pdt;  //2 - 0,(pos-1)*6+3  + sa transposee en (pos-1)*6+3,0
        ElMatrix <double> N1 = obs.transpose()*obs*pdt;  //3 - en (pos-1)*6+3, idem

        //1 - Ajout de C en (0,0) de _N
        for (int aK=0; aK < C.Sz().x; aK++)
            for (int bK=0; bK < C.Sz().y; bK++)
                _N(aK,bK) += C(aK,bK);

        //2 - Ajout de C1 et C1t

        if (verbose) cout << "cam->mIndex : " << pos << endl;

        const int numUnk_int = (int)numUnk;
        if (pos > 0)
        {
            if (verbose)  printMatrix(C1, "C1");

            for (int aK=0; aK < C1.Sz().x; aK++)
                for (int bK=0; bK < C1.Sz().y; bK++)
                    _N( (pos-1)*numUnk_int+3 + aK, bK) += C1(aK,bK);

            ElMatrix <double> C1t = C1.transpose();

            if (verbose)  printMatrix(C1t, "C1t");

            for (int aK=0; aK < C1t.Sz().x; aK++)
                for (int bK=0; bK < C1t.Sz().y; bK++)
                    _N(aK, (pos-1)*numUnk_int+3+ bK) += C1t(aK,bK);

            //3 - Ajout de N1

            if (verbose)  printMatrix(N1, "N1");

            for(int aK=0; aK < N1.Sz().x; aK++)
                for (int bK=0; bK < N1.Sz().y; bK++)
                    _N((pos-1)*numUnk_int+3+aK, (pos-1)*numUnk_int+3 + bK) += N1(aK,bK);
        }

        //pour Y
        ElMatrix <double> Y1 = obs.transpose()*res*pdt;
        ElMatrix <double> C2 = ccc.transpose()*res*pdt;

        for (int aK=0; aK<C2.Sz().y;++aK)
            _Y(0,aK) += C2(0,aK);

        if (pos > 0)
        for (int aK=0; aK<Y1.Sz().y;++aK)
            _Y(0,(pos-1)*numUnk_int+3+aK) += Y1(0,aK);

        if (verbose)
        {
            printMatrix(_N, "_N");
            printMatrix(_Y, "_Y");
        }
    }

    void addObs(const ElMatrix<double> &ccc, const double p, const double res)
    {
        bool verbose = _verbose;

        double pdt = 1./(p*p);

        ElMatrix <double> C  = ccc.transpose()*ccc*pdt;  //1 - ajouter en 0,0

        //1 - Ajout de C en (0,0) de _N
        for (int aK=0; aK < C.Sz().x; aK++)
            for (int bK=0; bK < C.Sz().y; bK++)
                _N(aK,bK) += C(aK,bK);

        //pour Y
        ElMatrix <double> C2 = ccc.transpose()*res*pdt;

        for (int aK=0; aK<C2.Sz().y;++aK)
            _Y(0,aK) += C2(0,aK);

        if (verbose)
        {
            printMatrix(_N, "_N");
            printMatrix(_Y, "_Y");
        }
    }

    void addObsStabil(int pos, const ElMatrix<double> &obs, const double p, const double res)
    {
        bool verbose = false;

        double pdt = 1./(p*p);

        ElMatrix <double> N1 = obs.transpose()*obs*pdt;  //3 - en (pos-1)*6+3, idem

        if (verbose) cout << "cam->mIndex : " << pos << endl;

        const int numUnk_int = (int)numUnk;
        if (pos > 0)
        {
            for(int aK=0; aK < N1.Sz().x; aK++)
                for (int bK=0; bK < N1.Sz().y; bK++)
                    _N((pos-1)*numUnk_int+3+aK, (pos-1)*numUnk_int+3 + bK) += N1(aK,bK);
        }

        //pour Y
        ElMatrix <double> Y1 = obs.transpose()*res*pdt;

        if (pos > 0)
        for (int aK=0; aK< Y1.Sz().y;++aK)
            _Y(0,(pos-1)*numUnk_int+3+aK) += Y1(0,aK);

        if (verbose)
        {
            printMatrix(_N, "_N");
            printMatrix(_Y, "_Y");
        }
    }

    void solveFirstGroup(vector<int> const &vpos)
    {
        bool verbose = false;

        if (verbose) cout << "solveFirstGroup : "  << endl;

        //matrice 3,3 pivot, l'inverser -> c-1 ElSubMat(0,0,3,3)
        ElMatrix <double> C = _N.sub_mat(0,0,3,3);

        if (verbose)   printMatrix(C, "C");

        ElMatrix <double> Cinv = gaussj(C);

        if (verbose)   printMatrix(Cinv, "Cinv");

        //matrice 3,1 pivotY, > Y0 ElSubMat(0,0,3,1)
        ElMatrix <double> Y0 = _Y.sub_mat(0,0,1,3);

        if (verbose)   printMatrix(Y0, "Y0");

        //pour ligne k : extraire ElSubMat(0,k*6+3,6,3) = Dk :
        //si = 0 rien, ligne suivante
        //sinon, extraire de Y sub(0,6*k..,6,1) puis Y -= Dk*C-1*Y0
        //       extraire pour tout col dans vPos sub(ligne k,vPos[k],6,6) puis N -= Dk*C-1*N(0,vPos[k],3,6)

        const int numUnk_int = (int)numUnk;
        for (size_t k =0; k < vpos.size(); ++k)
        {
            ElMatrix <double> Dk = _N.sub_mat(0, (vpos[k]-1)*numUnk_int+3, 3, numUnk_int);

            ElMatrix <double> M2 = Dk*Cinv*Y0;

            if (verbose)   printMatrix(M2, "M2");

            //Y -= Dk*C-1*Y0
            for (int bK=0; bK < M2.Sz().y; bK++)
                _Y(0, (vpos[k]-1)*numUnk_int+3+bK) -= M2(0,bK);

            if (verbose)
            {
                printMatrix(Dk, "Dk");

                printMatrix(Cinv, "Cinv");
            }

            for (size_t k2 =0; k2 < vpos.size(); ++k2)
            {
                ElMatrix <double> Dk2 = _N.sub_mat((vpos[k2]-1)*numUnk_int+3,0,numUnk_int,3);

                //printMatrix(Dk2, "Dk2");

                ElMatrix <double> N2 = Dk*Cinv*Dk2;

                //printMatrix(N2, "N2");

                // N -= Dk*C-1*N(0,vPos[k],3,6)
                for(int aK=0; aK < N2.Sz().x; aK++)
                    for (int bK=0; bK < N2.Sz().y; bK++)
                        _N((vpos[k2]-1)*numUnk_int+3+aK, (vpos[k]-1)*numUnk_int+3+bK) -= N2(aK, bK);
            }
        }

        //RAZ des 1eres lignes et colonnes (3)
        for(int aK=0; aK<_N.Sz().x;++aK)
        {
            for(int bK=0;bK <3;++bK)
                _N(aK,bK)=0;
        }

        for(int aK=0; aK<_N.Sz().y;++aK)
        {
            for(int bK=0;bK <3;++bK)
                _N(bK,aK)=0;
        }

        for (int aK=0; aK<3;++aK)
            _Y(0,aK) = 0;

        if (verbose)
        {
            printMatrix(_N, "_N");
            printMatrix(_Y, "_Y");
        }
    }

    void solve()
    {
        bool verbose = true;

        if (verbose) printMatrix(_N);

        ElMatrix<double> Nsub = _N.sub_mat(3,3,_N.Sz().x-3, _N.Sz().y-3 );
        ElMatrix<double> Ysub = _Y.sub_mat(0,3,1, _Y.Sz().y-3 );

        ElMatrix<double> inv = gaussj(Nsub);
        if (verbose)  printMatrix(inv, "inv");

        ElMatrix<double> sol = inv*Ysub;

        if (verbose) printMatrix(sol, "sol");

        //cout << "SOL_NORM = " << sol.NormC(2) << endl;



        //for (size_t aK=0; aK < vObs.size(); aK++)
        for (size_t aK=0; aK < 10; aK++)
        {
             Observation* aObs = vObs[aK];
             cout << aObs->getCoord().z << endl;
        }


        const int numUnk_int = (int)numUnk;
        int aK =0;
        map<int, AffCamera *>::const_iterator iter = mapCameras.begin();
        iter++; //don't use first image
        for (; iter != mapCameras.end();++iter)
        {
            ElMatrix <double> solSub = sol.sub_mat(0,aK,1,numUnk_int);
            iter->second->updateParams(solSub);

            aK += numUnk_int;
        }


        for (size_t aK=0; aK < 10; aK++)
        {
             Observation* aObs = vObs[aK];
             cout << aObs->getCoord().z << endl;
        }
    }

    //! compute the observation matrix for one iteration
    bool computeObservationMatrix()
    {
        cout <<"iter="<<iteration<<endl;

        double NoData = -9999.;

        int numObs;
        double res = sumRes(numObs);
        cout << "res= " << res << " numObs= " << numObs << endl;
        if (numObs)
        {
            double iniRMS = sqrt(res/numObs);
            //cout << "RMS_ini = " << iniRMS << endl;

            //double dA0 = 0.5;
            double dA1 = 0.01;
            double dA2 = 0.01;
            //double dB0 = 0.5;
            double dB1 = 0.01;
            double dB2 = 0.01;
            double dX = 0.1;
            double dY = 0.1;
            double dZ = 0.1;

            //Get number of cameras to estimate
            int nbCam = 0;
            map<int, AffCamera *>::const_iterator it = mapCameras.begin();
            for(;it !=mapCameras.end();++it)
                if (it->second->isActivated()) nbCam++;

            cout << "Nb cam to estimate : " << nbCam << endl;

            //Init matrix
            const int numUnk_int = (int)numUnk;
            int matSz = 3+numUnk_int*nbCam;

            _N = ElMatrix<double>(matSz, matSz);
            _Y = ElMatrix<double>(1, matSz);

            //pour chaque observation
            for (size_t aK=0; aK < nObs(); aK++)
            {
                Observation* aObs = vObs[aK];

                vector <ImageMeasure> vMes = aObs->vImgMeasure;
                vector <int> vPos;

                Pt3dr pt = aObs->getCoord();

                //pour chaque image où le point de liaison est vu
                for(size_t bK=0; bK < vMes.size();++bK)
                {
                    const int bK_int = (int)bK;
                    Pt2dr D = aObs->computeImageDifference(bK_int, pt);
                   // double ecart2 = square_euclid(D);

                    double pdt = 1.; //1./sqrt(1. + ecart2);

                    //todo : strategie d'elimination d'observations / ou ponderation

                    ElMatrix<double> obs(numUnk_int,1);
                    ElMatrix<double> ccc(3,1);

                    // estimation des derivees partielles
                    map<int, AffCamera *>::iterator iter = mapCameras.find(vMes[bK].imgIndex());
                    AffCamera* cam = iter->second;

                    if (cam->index() > 0) vPos.push_back(cam->index());

                    double a0 = cam->vP[0];
                    double a1 = cam->vP[1];
                    double a2 = cam->vP[2];
                    double b0 = cam->vP[3];
                    double b1 = cam->vP[4];
                    double b2 = cam->vP[5];

                    //Pt2dr vdA0 = Pt2dr(1./dA0,1./dA0) * (aObs->computeImageDifference(bK_int,pt,a0+dA0,a1,a2,b0,b1,b2)-D);
                    Pt2dr vdA1 = Pt2dr(1./dA1,1./dA1) * (aObs->computeImageDifference(bK_int,pt,a0,a1+dA1,a2,b0,b1,b2)-D);
                    Pt2dr vdA2 = Pt2dr(1./dA2,1./dA2) * (aObs->computeImageDifference(bK_int,pt,a0,a1,a2+dA2,b0,b1,b2)-D);
                    //Pt2dr vdB0 = Pt2dr(1./dB0,1./dB0) * (aObs->computeImageDifference(bK_int,pt,a0,a1,a2,b0+dB0,b1,b2)-D);
                    Pt2dr vdB1 = Pt2dr(1./dB1,1./dB1) * (aObs->computeImageDifference(bK_int,pt,a0,a1,a2,b0,b1+dB1,b2)-D);
                    Pt2dr vdB2 = Pt2dr(1./dB2,1./dB2) * (aObs->computeImageDifference(bK_int,pt,a0,a1,a2,b0,b1,b2+dB2)-D);

                    Pt2dr vdX = Pt2dr(1./dX,1./dX) * (aObs->computeImageDifference(bK_int,Pt3dr(pt.x+dX, pt.y, pt.z),a0,a1,a2,b0,b1,b2)-D);
                    Pt2dr vdY = Pt2dr(1./dY,1./dY) * (aObs->computeImageDifference(bK_int,Pt3dr(pt.x, pt.y+dY, pt.z),a0,a1,a2,b0,b1,b2)-D);
                    Pt2dr vdZ = Pt2dr(1./dZ,1./dZ) * (aObs->computeImageDifference(bK_int,Pt3dr(pt.x, pt.y, pt.z+dZ),a0,a1,a2,b0,b1,b2)-D);

                    if (cam->index() !=0)
                    {

                        //obs(0,0) = vdA0.x;
                        obs(1,0) = vdA1.x;
                        obs(2,0) = vdA2.x;

                        //obs(3,0) = vdB0.x;
                        obs(4,0) = vdB1.x;
                        obs(5,0) = vdB2.x;

                    }

                    ccc(0,0) = vdX.x;
                    ccc(1,0) = vdY.x;
                    ccc(2,0) = vdZ.x;

                    addObs(cam->index(), obs, ccc, pdt,-D.x);

                    if (cam->index() !=0)
                    {

                        //obs(0,0) = vdA0.y;
                        obs(1,0) = vdA1.y;
                        obs(2,0) = vdA2.y;

                        //obs(3,0) = vdB0.y;
                        obs(4,0) = vdB1.y;
                        obs(5,0) = vdB2.y;

                    }

                    ccc(0,0) = vdX.y;
                    ccc(1,0) = vdY.y;
                    ccc(2,0) = vdZ.y;

                    addObs(cam->index(), obs, ccc, pdt,-D.y);
                }

                solveFirstGroup(vPos);

                //Contrainte en Z
                if (_MntImg.get()!=NULL)
                {
                    Pt2dr ptMnt(pt.x, pt.y);
                    double zMnt = _MntImg->getr(ptMnt,NoData)*_MntOri.ResolutionAlti() + _MntOri.OrigineAlti();

                    if (zMnt == NoData)
                        cout << "Pas d'altitude trouvee pour le point : "<< pt.x<< " " << pt.y << endl;
                    else
                    {
                        double res = -pt.z + zMnt;

                        ElMatrix<double> ccd(3,1);
                        ccd(0,0) = 0;
                        ccd(1,0) = 0;
                        ccd(2,0) = 1;

                        double sigma = 20.;

                        addObs(ccd, sigma, res);
                    }
                }
            }

            map<int, AffCamera *>::iterator iter = mapCameras.begin();
            for(; iter!=mapCameras.end();++iter)
            {
                AffCamera* cam = iter->second;

                for (size_t aK=0; aK < numUnk;aK++)
                {
                    ElMatrix <double> AB (numUnk_int, 1);
                    AB((int)aK,0) = 1.;

                    double sig = 1.;
                    if ((aK==0) || (aK==3)) sig = 0.1;  //pix
                        else sig = 1e-4;

                    /*if ((aK==1) || (aK==5))
                        addObsStabil(cam->mIndex, AB, sig, 1. - cam->vP[aK]);*/
                    /*else
                        addObsStabil(cam->mIndex, AB, sig, 0. - cam->vP[aK]);*/

                     if ((aK==0) || (aK==3))
                        addObsStabil(cam->index(), AB, sig, 0. - cam->vP[aK]);
                }
            }

            cout << "before solve"<<endl;

            solve();

            return launchNewIter(iniRMS, numObs);
        }
        else
        {
            cout << "Error in computeObservationMatrix numObs=0" << endl;
            return false;
        }
    }

    ~RefineModelGlobal()
    {
    }

private:
    std_unique_ptr<TIm2D<REAL4,REAL8> > _MntImg;
    cFileOriMnt                        _MntOri;
};

int NewRefineModel_main(int argc, char **argv)
{
    string aPat; // GRID files pattern
    string aImgsExtension=".tif"; //img file extension (such used in Tapioca)
    string aNameMNT=""; //DTM file
    string aNamePointeIm=""; //Pointe image file
    string aNameGCP=""; //GCP file
    bool filterInput=false;
    bool exportResidus = false;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aPat,"GRID files pattern")
                    << EAMC(aImgsExtension, "Img files extension"),
         LArgMain() << EAM(aNameGCP,"GCP",true, "GCP file")
                    << EAM(aNamePointeIm,"IMG",true, "Pointe image file")
                    << EAM(aNameMNT,"DTM",true, "DTM file")
                    << EAM(filterInput,"Filter",true, "Remove tiepoints with ground distance > 10m (def=false)")
                    << EAM(exportResidus,"ExpRes",true, "Export residuals (def=false)")
    );

    ELISE_fp::MkDirSvp("refine");

    RefineModelGlobal model(aPat, aImgsExtension, aNameGCP, aNamePointeIm, aNameMNT, filterInput);

    bool ok = (model.nObs() > 3);
    for(size_t iter = 0; (iter < 100) & ok; iter++)
        ok = model.computeObservationMatrix();

    if (exportResidus)
    {
        ofstream ficRes("refine/residus.txt");
        ofstream ficGlb("refine/residusGlob.txt");
        ficRes << setprecision(15);
        ficGlb << setprecision(15);

        vector <Observation*> vTP = model.getObs();
        for (size_t aK=0; aK < vTP.size();++aK)
        {
            Observation* aObs = vTP[aK];

            Pt3dr PT = aObs->getCoord();

            double sumRes = 0.;
            for(size_t i=0;i<aObs->vImgMeasure.size();++i)
            {
                Pt2dr D = aObs->computeImageDifference((int)i, PT);

                ficRes << aK <<" "<< aObs->vImgMeasure[i].imgName() <<" "<< aObs->vImgMeasure[i].pt().x <<" "<< aObs->vImgMeasure[i].pt().y <<" "<< D.x <<" "<< D.y <<" "<<endl;

                sumRes += square_euclid(D);
            }

            ficGlb << aK <<" "<< PT.x << " " << PT.y << " " << PT.z << " "<< sumRes << " " << sqrt(sumRes/aObs->vImgMeasure.size()) << endl;
        }
    }

    return EXIT_SUCCESS;
}

int AddAffinity_main(int argc, char **argv)
{
    string aNameFile; // tie-points file
    bool addNoise = false;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aNameFile,"tie-points file"),
         LArgMain() << EAM(addNoise, "Noise", true, "Add noise (def=false)" )
    );

    vector <DigeoPoint> aList;
    DigeoPoint::readDigeoFile(aNameFile, true, aList);
    //TODO: handle versions

    cout << "pts nb: " << aList.size() << endl;

    double A0, A1, A2, B0, B1, B2;
    A0 = 0.1f;
    A1 = 0.999999985;
    A2 = -0.000174533;
    B0 = 0.05;
    B1 = 0.000174533;
    B2 = 0.999999985;

    for (size_t aK = 0 ; aK< aList.size(); aK++)
    {
        DigeoPoint pt = aList[aK];

        //cout << "pt avt " << pt.x << " " << pt.y << endl;
        Pt2dr ptt(A0 + A1 * pt.x + A2 * pt.y, B0 + B1 * pt.x + B2 * pt.y);

        if (addNoise)
        {
            Pt2dr aNoise(NRrandC(),NRrandC());
            ptt += aNoise;
        }

        //cout << "pt apr " << ptt.x << " " << ptt.y << endl;

        aList[aK].x = ptt.x;
        aList[aK].y = ptt.y;
    }

    string aOut = StdPrefixGen(aNameFile)+ "_transf.dat";
    DigeoPoint::writeDigeoFile(aOut, aList, 0, true);

    return EXIT_SUCCESS;
}



///// ****** OLD CODE TO KEEP COMPATIBILITY WITH DOC *******
///
///

class OldAffCamera
{
 public:
    OldAffCamera(string aFilename):a0(0),a1(1),a2(0),b0(0),b1(0),b2(1)
    {
        // Loading the GRID file
        ElAffin2D oriIntImaM2C;
        Pt2di Sz(10000,10000);
        mCamera =  new cCameraModuleOrientation(new OrientationGrille(aFilename),Sz,oriIntImaM2C);
    }

    ///
    /// \brief Tie Points (in pixel)
    ///
    vector<Pt2dr> vPtImg;

    ///
    /// \brief update affinity parameters
    /// \param sol unknowns matrix
    ///
    void updateParams(ElMatrix <double> const &sol)
    {
        cout << "Init solution: "<<endl;
        printParams();

        a0 += sol(0,0);
        a1 += sol(0,1);
        a2 += sol(0,2);
        b0 += sol(0,3);
        b1 += sol(0,4);
        b2 += sol(0,5);

        cout << "Updated solution: "<<endl;
        printParams();
    }

    void printParams()
    {
        cout << a0<<" "<<a1<<" "<<a2<<endl;
        cout << b0<<" "<<b1<<" "<<b2<<endl;
    }

    ElCamera* Camera() { return mCamera; }

    // the 6 parameters of affinity
    // colc = a0 + a1 * col + a2 * lig
    // ligc = b0 + b1 * col + b2 * lig
    double a0;
    double a1;
    double a2;
    double b0;
    double b1;
    double b2;

protected:

    ElCamera* mCamera;
};

//! Abstract class for shared methods
class OldRefineModelAbs
{
protected:
    OldAffCamera* master;
    OldAffCamera* slave;

    ///
    /// \brief Points altitude (to estimate)
    ///
    vector<double> vZ;
    ///
    /// \brief zMoy Ground mean altitude
    ///
    double zMoy;

    ///
    /// \brief normal matrix for least squares estimation
    ///
    ElMatrix<double> _N;
    ///
    /// \brief matrix for least squares estimation
    ///
    ElMatrix<double> _Y;

public:

    ///
    /// \brief Z estimation (iterative: 2D ground distance minimization)
    /// \param ptImgMaster tie-point from master image
    /// \param ptImgSlave  tie-point from slave image
    /// \param zIni init altitude
    /// \param dZ shift on altitude
    /// \return Z altitude of tie-point
    ///
    double getZ(Pt2dr const &ptImgMaster,
                Pt2dr const &ptImgSlave,
                double zIni,
                double dZ = 0.1) const
    {
        double z = zIni;
        Pt2dr D   = compute2DGroundDifference(ptImgMaster,ptImgSlave,z, slave);
        double d  = square_euclid(D);
        Pt2dr D1  = compute2DGroundDifference(ptImgMaster,ptImgSlave,z-dZ, slave);
        double d1 = square_euclid(D1);
        Pt2dr D2  = compute2DGroundDifference(ptImgMaster,ptImgSlave,z+dZ, slave);
        double d2 = square_euclid(D2);
        if (d1<d2)
        {
            while(d1<d)
            {
                d = d1;
                z = z-dZ;
                D1 = compute2DGroundDifference(ptImgMaster,ptImgSlave,z-dZ, slave);
                d1 = square_euclid(D1);
            }
        }
        else
        {
            while(d2<d)
            {
                d = d2;
                z = z+dZ;
                D2 = compute2DGroundDifference(ptImgMaster,ptImgSlave,z+dZ, slave);
                d2 = square_euclid(D2);
            }
        }
        return z;
    }

    ///
    /// \brief constructor (loads GRID files, tie-points and filter tie-points on 2D ground difference)
    /// \param aNameFileGridMaster Grid file for master image
    /// \param aNameFileGridSlave Grid file for slave image
    /// \param aNamefileTiePoints Tie-points file
    /// \param Zmoy ground mean altitude
    ///
    OldRefineModelAbs(string const &aNameFileGridMaster,
                   string const &aNameFileGridSlave,
                   string const &aNameFileTiePoints,
                   double Zmoy):master(NULL),slave(NULL),zMoy(Zmoy),_N(1,1,0.),_Y(1,1,0.)
    {
        // Loading the GRID file
        master = new OldAffCamera(aNameFileGridMaster);
        slave  = new OldAffCamera(aNameFileGridSlave);

        // Loading the Tie Points
        ElPackHomologue aPackHomol = ElPackHomologue::FromFile(aNameFileTiePoints);

        if (aPackHomol.size() == 0)
        {
            cout << "Error in RefineModelAbs: no tie-points" << endl;
            return;
        }

        int rPts_nb = 0; //rejected points number

        ElPackHomologue::const_iterator iter = aPackHomol.begin();
        for (;iter!=aPackHomol.end();++iter )
        {
            ElCplePtsHomologues aCple = iter->ToCple();

            Pt2dr P1 = aCple.P1();
            Pt2dr P2 = aCple.P2();

            double z = getZ(P1,P2,zMoy);
            //cout << "z = "<<z<<endl;
            Pt2dr D = compute2DGroundDifference(P1,P2,z,slave);

            if (square_euclid(D)>100.)
            {
                rPts_nb++;
                //cout << "Point with 2D ground difference > 10 : "<< D.x << " " << D.y << " - rejected" << endl;
            }
            else
            {
                master->vPtImg.push_back(P1);
                slave->vPtImg.push_back(P2);
                vZ.push_back(z);
            }
        }

        cout << "Number of rejected points : "<< rPts_nb << endl;
        cout << "Number of tie points : "<< master->vPtImg.size() << endl;
    }

    ///
    /// \brief compute the difference between the Ground Points for a given Tie Point and a given set of parameters (Z and affinity)
    /// \param ptImgMaster tie-point from master image
    /// \param ptImgSlave tie-point from slave image
    /// \param aZ   ground altitude
    /// \param aA0  affinity parameter
    /// \param aA1  affinity parameter
    /// \param aA2  affinity parameter
    /// \param aB0  affinity parameter
    /// \param aB1  affinity parameter
    /// \param aB2  affinity parameter
    /// \return Pt2Dr 2D difference between ground points
    ///
    Pt2dr compute2DGroundDifference(Pt2dr const &ptImgMaster,
                                    Pt2dr const &ptImgSlave,
                                    double aZ,
                                    double aA0,
                                    double aA1,
                                    double aA2,
                                    double aB0,
                                    double aB1,
                                    double aB2)const
    {
        Pt3dr ptTerMaster = master->Camera()->ImEtProf2Terrain(ptImgMaster,aZ);
        Pt2dr ptImgSlaveC(aA0 + aA1 * ptImgSlave.x + aA2 * ptImgSlave.y,
                          aB0 + aB1 * ptImgSlave.x + aB2 * ptImgSlave.y);
        Pt3dr ptTerSlave = slave->Camera()->ImEtProf2Terrain(ptImgSlaveC,aZ);
        return Pt2dr(ptTerMaster.x - ptTerSlave.x,ptTerMaster.y - ptTerSlave.y);
    }

    Pt2dr compute2DGroundDifference(Pt2dr const &ptImgMaster,
                                    Pt2dr const &ptImgSlave,
                                    double aZ, OldAffCamera* cam)const
    {
        return compute2DGroundDifference(ptImgMaster,ptImgSlave,aZ,cam->a0,cam->a1,cam->a2,cam->b0,cam->b1,cam->b2);
    }

    ///
    /// \brief 2D ground distance sum for all tie points (to compute RMS)
    /// \return sum of residuals
    ///
    double sumRes()
    {
        double sumRes = 0.;
        for(size_t i=0;i<master->vPtImg.size();++i)
        {
            Pt2dr const &ptImgMaster = master->vPtImg[i];
            Pt2dr const &ptImgSlave  = slave->vPtImg[i];
            // ecart constate
            Pt2dr D = compute2DGroundDifference(ptImgMaster,ptImgSlave,vZ[i], slave);
            sumRes += square_euclid(D);
        }
        return sumRes;
    }



    ///
    /// \brief debug matrix
    /// \param mat matrix to write
    ///
    void printMatrix(ElMatrix <double> const & mat)
    {
        cout << "-------------------------"<<endl;
        for(int i=0;i<mat.Sz().x;++i)
        {
            for(int j=0;j<mat.Sz().y;++j)
                cout << mat(i,j) <<" ";

            cout << endl;
        }
        cout << "-------------------------"<<endl;
    }

    ///
    /// \brief check if a new iteration should be run and write result file (at the step before exiting loop)
    /// \param iniRMS rms before system solve
    /// \param numObs system number of observations
    /// \return
    ///
    bool launchNewIter(double iniRMS, int numObs)
    {
        double curRMS = sqrt(sumRes()/numObs);

        if (curRMS>iniRMS)
        {
            cout << "curRMS = "<<curRMS<<" / iniRMS = "<<iniRMS<<endl;
            cout << "No improve: end"<<endl;
            return false;
        }

        //ecriture dans un fichier des coefficients en vue d'affiner la grille

        ofstream fic("refine/refineCoef.txt");
        fic << setprecision(15);
        fic << slave->a0 <<" "<< slave->a1 <<" "<< slave->a2 <<" "<< slave->b0 <<" "<< slave->b1 <<" "<< slave->b2 <<" "<<endl;
        cout << "RMS_after = " << curRMS << endl;
        return true;
    }

    ///
    /// \brief estimates affinity parameters
    ///
    virtual void solve()=0;

    ///
    /// \brief computes the observation matrix for one iteration
    /// \return boolean stating if system is solved (need new iteration)
    ///
    virtual bool computeObservationMatrix()=0;

    virtual ~OldRefineModelAbs()
    {
        if (master)
            delete master;
        if (slave)
            delete slave;
    }
};

//! Implementation basique (sans suppression des inconnues auxiliaires)
class OldRefineModelBasicSansZ: public OldRefineModelAbs
{

public:
    OldRefineModelBasicSansZ(string const &aNameFileGridMaster,
                     string const &aNameFileGridSlave,
                     string const &aNamefileTiePoints,
                     double Zmoy):OldRefineModelAbs(aNameFileGridMaster,aNameFileGridSlave,aNamefileTiePoints,Zmoy)
    {
    }

    void solve()
    {
        /*
         cout << "solve"<<endl;
         cout << "Matrice _N:"<<endl;
         printMatrix(_N);
         cout << "Matrice _Y:"<<endl;
         printMatrix(_Y);
         */
        ElMatrix<double> AtA = _N.transpose() * _N;
        // printMatrix(AtA);

        ElMatrix<double> AtB = _N.transpose() * _Y;
        //printMatrix(AtB);
        ElMatrix<double> sol = gaussj(AtA) * AtB;
        /*
         cout << "Matrice sol:"<<endl;
         printMatrix(sol);
         */
        //cout << "SOL_NORM = " << sol.NormC(2) << endl;

        slave->updateParams(sol);

        // Z update
        for(size_t i=0;i<master->vPtImg.size();++i)
        {
            vZ[i] = getZ(master->vPtImg[i],slave->vPtImg[i],vZ[i]);
           // vZ[i] = getZ(vPtImgMaster[i],vPtImgSlave[i],vZ[i], 0.01); => legere amelioration
        }
    }

    //! compute the observation matrix for one iteration
    bool computeObservationMatrix()
    {
        int numObs = 2*(int)vZ.size();
        double iniRMS = sqrt(sumRes()/numObs);
        cout << "RMS_ini = " << iniRMS << endl;

        double dA0 = 0.5;
        double dA1 = 0.01;
        double dA2 = 0.01;
        double dB0 = 0.5;
        double dB1 = 0.01;
        double dB2 = 0.01;

        _N = ElMatrix<double>(6,2*(int)vZ.size()/*+6*/,0.);
        _Y = ElMatrix<double>(1,2*(int)vZ.size()/*+6*/,0.);

        //pour chaque obs (ligne), y compris les eq de stabilisation
        //for( toutes les obs )
        for(size_t i=0;i<master->vPtImg.size();++i)
        {
            const int i_int = (int)i;
            //cout << "i = "<<i<<endl;
            Pt2dr const &ptImgMaster = master->vPtImg[i];
            Pt2dr const &ptImgSlave  = slave->vPtImg[i];
            double const Z = vZ[i];

            // ecart constate
            Pt2dr D = compute2DGroundDifference(ptImgMaster,ptImgSlave,Z, slave);
            double ecart2 = square_euclid(D);

            double pdt = 1./sqrt(1. + ecart2);

            //todo : strategie d'elimination d'observations / ou ponderation

            // estimation des derivees partielles
            double a0 = slave->a0;
            double a1 = slave->a1;
            double a2 = slave->a2;
            double b0 = slave->b0;
            double b1 = slave->b1;
            double b2 = slave->b2;

            Pt2dr vdA0 = Pt2dr(1./dA0,1./dA0) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0+dA0,a1,a2,b0,b1,b2)-D);
            Pt2dr vdA1 = Pt2dr(1./dA1,1./dA1) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1+dA1,a2,b0,b1,b2)-D);
            Pt2dr vdA2 = Pt2dr(1./dA2,1./dA2) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2+dA2,b0,b1,b2)-D);
            Pt2dr vdB0 = Pt2dr(1./dB0,1./dB0) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0+dB0,b1,b2)-D);
            Pt2dr vdB1 = Pt2dr(1./dB1,1./dB1) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1+dB1,b2)-D);
            Pt2dr vdB2 = Pt2dr(1./dB2,1./dB2) * (compute2DGroundDifference(ptImgMaster,ptImgSlave,Z,a0,a1,a2,b0,b1,b2+dB2)-D);

            _N(0,2*i_int) = pdt * vdA0.x;
            _N(1,2*i_int) = pdt * vdA1.x;
            _N(2,2*i_int) = pdt * vdA2.x;
            _N(3,2*i_int) = pdt * vdB0.x;
            _N(4,2*i_int) = pdt * vdB1.x;
            _N(5,2*i_int) = pdt * vdB2.x;

            _N(0,2*i_int+1) = pdt * vdA0.y;
            _N(1,2*i_int+1) = pdt * vdA1.y;
            _N(2,2*i_int+1) = pdt * vdA2.y;
            _N(3,2*i_int+1) = pdt * vdB0.y;
            _N(4,2*i_int+1) = pdt * vdB1.y;
            _N(5,2*i_int+1) = pdt * vdB2.y;

            _Y(0,2*i_int)   = pdt * (0.-D.x);
            _Y(0,2*i_int+1) = pdt * (0.-D.y);
        }
        // Equation de stabilisation
        /*
        {
            double pdt = vZ.size()/100.;
            // A0 proche de 0
            _N(0,2*vZ.size()) = 1 * pdt;
            _Y(0,2*vZ.size()) = (0-a0) * pdt;
            // A1 proche de 1
            _N(1,2*vZ.size()+1) = 1 * pdt;
            _Y(0,2*vZ.size()+1) = (1-a1) * pdt;
            // A2 proche de 0
            _N(2,2*vZ.size()+2) = 1 * pdt;
            _Y(0,2*vZ.size()+2) = (0-a2) * pdt;
            // B0 proche de 0
            _N(3,2*vZ.size()+3) = 1 * pdt;
            _Y(0,2*vZ.size()+3) = (0-b0) * pdt;
            // B1 proche de 0
            _N(4,2*vZ.size()+4) = 1 * pdt;
            _Y(0,2*vZ.size()+4) = (0-b1) * pdt;
            // B2 proche de 1
            _N(5,2*vZ.size()+5) = 1 * pdt;
            _Y(0,2*vZ.size()+5) = (1-b2) * pdt;
        }
         */
        cout << "before solve"<<endl;

        solve();

        return launchNewIter(iniRMS, numObs);
    }

    ~OldRefineModelBasicSansZ()
    {
    }
};

int RefineModel_main(int argc, char **argv)
{
    string aNameFileGridMaster; // fichier GRID image maitre
    string aNameFileGridSlave;  // fichier GRID image secondaire
    string aNameFileTiePoints;  // fichier de points de liaison
    double aZMoy;                    // the average altitude of the TiePoints

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aNameFileGridMaster,"master image GRID")
                    << EAMC(aNameFileGridSlave,"slave image GRID")
                    << EAMC(aNameFileTiePoints,"Tie Points")
                    << EAMC(aZMoy,"average altitude of the TiePoints"),
         LArgMain()
    );

    ELISE_fp::MkDirSvp("refine");

    OldRefineModelBasicSansZ model(aNameFileGridMaster,aNameFileGridSlave,aNameFileTiePoints,aZMoy);

    bool ok=true;
    for(size_t iter = 0; (iter < 100) & ok; iter++)
    {
        cout <<"iter="<<iter<<endl;
        ok = model.computeObservationMatrix();
    }

    return EXIT_SUCCESS;
}


