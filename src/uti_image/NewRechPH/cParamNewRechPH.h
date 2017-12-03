#ifndef Define_NotRechNewPH
#define Define_NotRechNewPH
// NOMORE ...
class cPCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPCarac & anObj,cElXMLTree * aTree);


        Pt2dr & Pt();
        const Pt2dr & Pt()const ;
    private:
        Pt2dr mPt;
};
cElXMLTree * ToXMLTree(const cPCarac &);

void  BinaryDumpInFile(ELISE_fp &,const cPCarac &);

void  BinaryUnDumpFromFile(cPCarac &,ELISE_fp &);

std::string  Mangling( cPCarac *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotRechNewPH
