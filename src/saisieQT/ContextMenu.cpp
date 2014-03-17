#include "ContextMenu.h"

void ContextMenu::createContextMenuActions()
{
    QString IconFolder = QString(MMDir().c_str()) + "data/ico/";

    _highLight = new QAction(QIcon(IconFolder + "HL.ico"),              tr("Highlight"), this);

    _AllW      = new QAction(tr("All Windows") , this);
    _ThisW     = new QAction(tr("This Window"), this);
    _RollW     = new QAction(tr("Roll Window"), this);
    _ThisP     = new QAction(tr("This Point"), this);

    _validate  = new QAction(QIcon(IconFolder + "smile.ico"),           tr("Validate"), this);
    _dubious   = new QAction(QIcon(IconFolder + "interrogation.ico"),   tr("Dubious") , this);
    _refuted   = new QAction(QIcon(IconFolder + "refuted.ico"),         tr("Refuted") , this);
    _noSaisie  = new QAction(QIcon(IconFolder + "vide.ico"),            tr("Not captured"), this);

    _rename    = new QAction(tr("Rename"), this);

    connect(_rename,		    SIGNAL(triggered()),   this, SLOT(rename()));

    connect(_highLight,		    SIGNAL(triggered()),   this, SLOT(highlight()));

    _switchSignalMapper = new QSignalMapper (this);

    connect(_AllW,      	    SIGNAL(triggered()),   _switchSignalMapper, SLOT(map()));
    connect(_RollW,      	    SIGNAL(triggered()),   _switchSignalMapper, SLOT(map()));
    connect(_ThisW,             SIGNAL(triggered()),   _switchSignalMapper, SLOT(map()));
    connect(_ThisP,             SIGNAL(triggered()),   _switchSignalMapper, SLOT(map()));

    _switchSignalMapper->setMapping (_AllW,  eAllWindows);
    _switchSignalMapper->setMapping (_ThisW, eThisWindow);
    _switchSignalMapper->setMapping (_ThisP, eThisPoint);
    _switchSignalMapper->setMapping (_RollW, eThisWindow);

    connect (_switchSignalMapper, SIGNAL(mapped(int)), this, SLOT(changeImages(int)));

    _stateSignalMapper = new QSignalMapper (this);

    connect(_validate,		    SIGNAL(triggered()),   _stateSignalMapper, SLOT(map()));
    connect(_dubious,		    SIGNAL(triggered()),   _stateSignalMapper, SLOT(map()));
    connect(_refuted,		    SIGNAL(triggered()),   _stateSignalMapper, SLOT(map()));
    connect(_noSaisie,		    SIGNAL(triggered()),   _stateSignalMapper, SLOT(map()));

    _stateSignalMapper->setMapping (_validate,  eEPI_Valide);
    _stateSignalMapper->setMapping (_dubious,   eEPI_Douteux);
    _stateSignalMapper->setMapping (_refuted,   eEPI_Refute);
    _stateSignalMapper->setMapping (_noSaisie,  eEPI_NonSaisi);

    connect (_stateSignalMapper, SIGNAL(mapped(int)), this, SLOT(setPointState(int)));
}

void ContextMenu::setPointState(int state)
{
    int idx = _polygon->setNearestPointState(_lastPosImage, state);

    emit changeState(state, idx);
}

void ContextMenu::changeImages(int mode)
{
    int idx = -4;
    bool aUseCpt = false;

    switch(mode)
    {
    case eAllWindows:
        break;
    case eThisWindow:
        idx = -2;
        break;
    case eThisPoint:
        idx = _polygon->getNearestPointIndex(_lastPosImage);
        break;
    case eRollWindow:
        aUseCpt= true;
        break;
    }

    emit changeImagesSignal(idx, aUseCpt);
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

    QString oldName = _polygon->getNearestPointName(_lastPosImage);

    QString newName = inputDialog->getText(NULL, tr("Rename"), tr("Point name:"), QLineEdit::Normal, oldName);

    if (!newName.isEmpty())

        emit changeName(oldName, newName);
}
