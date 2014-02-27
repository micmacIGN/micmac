#include "ContextMenu.h"

void ContextMenu::createContextMenuActions()
{
    QString IconFolder = QString(MMDir().c_str()) + "data/ico/";

    _highLight = new QAction(QIcon(IconFolder + "HL.ico"),              tr("Highlight"), this);

    _AllW      = new QAction(tr("All Windows") , this);
    _ThisW     = new QAction(tr("This Window"), this);
    _ThisP     = new QAction(tr("This Point"), this);

    _validate  = new QAction(QIcon(IconFolder + "smile.ico"),           tr("Validate"), this);
    _dubious   = new QAction(QIcon(IconFolder + "interrogation.ico"),   tr("Dubious") , this);
    _refuted   = new QAction(QIcon(IconFolder + "refuted.ico"),         tr("Refuted") , this);
    _noSaisie  = new QAction(QIcon(IconFolder + "vide.ico"),            tr("Not captured"), this);

    _rename    = new QAction(tr("Rename"), this);

    connect(_rename,		    SIGNAL(triggered()),   this, SLOT(rename()));

    connect(_highLight,		    SIGNAL(triggered()),   this, SLOT(highlight()));

    _signalMapper = new QSignalMapper (this);

    /*connect(_AllW,      	    SIGNAL(triggered()),   _signalMapper, SLOT(AllW()));
    connect(_ThisW,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisW()));
    connect(_ThisP,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisP()));*/

    connect(_validate,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_dubious,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_refuted,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_noSaisie,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    _signalMapper->setMapping (_validate,  eEPI_Valide);
    _signalMapper->setMapping (_dubious,   eEPI_Douteux);
    _signalMapper->setMapping (_refuted,   eEPI_Refute);
    _signalMapper->setMapping (_noSaisie,  eEPI_NonSaisi);

    connect (_signalMapper, SIGNAL(mapped(int)), this, SLOT(setPointState(int)));
}

void ContextMenu::setPointState(int state)
{
    int idx = _polygon->setNearestPointState(_lastPosImage, state);

    emit    changeState(state, idx);
}

void ContextMenu::highlight()
{
    int idx = _polygon->highlightNearestPoint(_lastPosImage);

    emit changeState(NS_SaisiePts::eEPI_Highlight, idx);
}

void ContextMenu::rename()
{
    QInputDialog* inputDialog = new QInputDialog();
    inputDialog->setOptions(QInputDialog::NoButtons);

    QString text = inputDialog->getText(NULL, tr("Rename"), tr("Point name:"), QLineEdit::Normal, _polygon->getNearestPointName(_lastPosImage));

    if (!text.isEmpty())

         _polygon->rename(_lastPosImage, text);
}
