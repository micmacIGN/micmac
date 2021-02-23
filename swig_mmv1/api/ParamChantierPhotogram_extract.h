//extracted from XML_GEN/ParamChantierPhotogram.h
#ifndef PARAMCHANTIERPHOTOGRAM_EXTRACT_H
#define PARAMCHANTIERPHOTOGRAM_EXTRACT_H

class cSauvegardeNamedRel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSauvegardeNamedRel & anObj,cElXMLTree * aTree);


        std::vector< cCpleString > & Cple();
        const std::vector< cCpleString > & Cple()const ;
    private:
        std::vector< cCpleString > mCple;
};

#endif //PARAMCHANTIERPHOTOGRAM_EXTRACT_H
