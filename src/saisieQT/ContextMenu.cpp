#include "ContextMenu.h"

void ContextMenu::createContexMenuActions()
{
    QString IconFolder = QString(MMDir().c_str()) + "data/ico/";

    _rename    = new QAction(tr("Rename"), this);
    _showNames = new QAction(tr("Show names") , this);

    _highLight = new QAction(QIcon(IconFolder + "HL.ico"),              tr("Highlight"), this);

    _AllW      = new QAction(tr("AllW") , this);
    _ThisW     = new QAction(tr("ThisW"), this);
    _ThisP     = new QAction(tr("ThisP"), this);

    _validate  = new QAction(QIcon(IconFolder + "smile.ico"),           tr("Validate"), this);
    _dubious   = new QAction(QIcon(IconFolder + "interrogation.ico"),   tr("Dubious") , this);
    _refuted   = new QAction(QIcon(IconFolder + "refuted.ico"),         tr("Refuted") , this);
    _noSaisie  = new QAction(QIcon(IconFolder + "vide.ico"),            tr("Not captured"), this);

    connect(_rename,		    SIGNAL(triggered()),   this, SLOT(rename()));
    connect(_showNames,		    SIGNAL(triggered()),   this, SLOT(showNames()));

    connect(_highLight,		    SIGNAL(triggered()),   this, SLOT(highlight()));

    _signalMapper = new QSignalMapper (this);

    /*connect(_AllW,      	    SIGNAL(triggered()),   _signalMapper, SLOT(AllW()));
    connect(_ThisW,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisW()));
    connect(_ThisP,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisP()));*/

    connect(_validate,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_dubious,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_refuted,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_noSaisie,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    _signalMapper->setMapping (_validate,  NS_SaisiePts::eEPI_Valide);
    _signalMapper->setMapping (_dubious,   NS_SaisiePts::eEPI_Douteux);
    _signalMapper->setMapping (_refuted,   NS_SaisiePts::eEPI_Refute);
    _signalMapper->setMapping (_noSaisie,  NS_SaisiePts::eEPI_NonSaisi);

    connect (_signalMapper, SIGNAL(mapped(int)), this, SLOT(setPointState(int)));
}

void ContextMenu::setPointState(int state)
{
    _polygon->setNearestPointState(_lastPosImage, state);
}

void ContextMenu::highlight()
{
    _polygon->highlightNearestPoint(_lastPosImage);
}

void ContextMenu::rename()
{
    QInputDialog* inputDialog = new QInputDialog();
    inputDialog->setOptions(QInputDialog::NoButtons);

    QString name = _polygon->getNearestPointName(_lastPosImage);
    if (name == "") name = _polygon->getDefaultName();

    QString text =  inputDialog->getText(NULL ,"Rename", "Point name:", QLineEdit::Normal, name);

    if (!text.isEmpty())

         _polygon->rename(_lastPosImage, text);
}

void ContextMenu::showNames()
{
    _polygon->showNames();
}
