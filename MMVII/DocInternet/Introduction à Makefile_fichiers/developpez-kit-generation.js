/*******************************************************************************
 * Auteurs : developpez.com (Atomya Rise, djibril, Torgar...)
 * But     : JS des articles de developpez.com
 * Date    : 03/08/2015
 ******************************************************************************/

/**********************************************************************************************************************************/
/******************************************* VARIABLES GLOBALES *******************************************************************/
/**********************************************************************************************************************************/
var hauteurSommaire, largeurSommaire, hauteurDroiteSommaire, margeBas, classNavSom, classFixeMarge, classInfoAuth, classToggle, TogglePlier, ToggleDeplier, hauteurContenuArticle, verifPresenceFixeMarge, verifPresenceInfoAuth, articleBody, barreNavigation, LargeurEcranMobile;
var cheminImageKit = "https://www.developpez.com/template/kit/";

/**********************************************************************************************************************************/
/******************************************* DOCUMENT READY ***********************************************************************/
/**********************************************************************************************************************************/
jQuery(function($) {

	/******************************************* On initialise les variables **********************************************************/
	classNavSom            = jQuery(".nav-sommaire");                    // Section du sommaire
	classFixeMarge         = jQuery(".fixeMarge");                       // Div qui contient le synopsis
	classInfoAuth          = jQuery(".InfoAuthArtBook");                 // Section contenant les auteurs, info et bookmark
	classToggle            = jQuery(".ToggleGecko");                     // Entete sommaire
	TogglePlier            = jQuery(".ToggleImgPlier");
	ToggleDeplier          = jQuery(".ToggleImgDeplier");
	verifPresenceFixeMarge = jQuery(".fixeMarge").length;
	verifPresenceInfoAuth  = jQuery(".InfoAuthArtBook").length;
	articleBody            = jQuery(".articleBody");                     // Corps de l'article
	barreNavigation        = jQuery(".barreNavigation");
	hauteurSommaire        = classNavSom.height();
	margeBas               = 160 + jQuery("#gabarit_pied").height();
	hauteurContenuArticle  = articleBody.height();

	/* Ecran mobile */
	LargeurEcranMobile     = 975;
	var LargeurEcran = $( window ).width();
	var LargeurEcranVideo = LargeurEcran - 50;

	/******************************************* Gestion des largeurs des sections d'entête *************************************************/
	/* 32%, 48%, 15%*/
	var LargeurSectionAuteurs            = jQuery(".SectionAuteurs").css("width");
	var LargeurSectionInformationArticle = jQuery(".SectionInformationArticle").css("width");
	var LargeurSectionBookmarks          = jQuery(".SectionBookmarks").css("width");

	if ( LargeurEcran > LargeurEcranMobile ) { 
		if ( ! LargeurSectionBookmarks && (LargeurSectionAuteurs && LargeurSectionInformationArticle )  ) {
			jQuery(".SectionInformationArticle").css("width", "63%");
		}
		else if  ( ! LargeurSectionAuteurs && (LargeurSectionBookmarks && LargeurSectionInformationArticle )  ) {
			jQuery(".SectionInformationArticle").css("width", "80%");
		}
		else if  ( ! LargeurSectionAuteurs &&  ! LargeurSectionBookmarks ) {
			jQuery(".SectionInformationArticle").css("width", "100%");
		}
	}

	/******************************************* On récupère la position du sommaire *************************************************/
	var positionSommaire;
	// S'il y a un sommaire, on récupère la position.
	if (classNavSom.length != 0) {
		var positionSommaire = classNavSom.offset();
		positionSommaire     = positionSommaire.top;
	}

	/******************************************* On fixe la hauteur de classInfoAuth *************************************************/
	var SectionAuteurs            = jQuery(".SectionAuteurs").height();
	var SectionInformationArticle = jQuery(".SectionInformationArticle").height();
	var SectionBookmarks          = jQuery(".SectionBookmarks").height();
	var plusGrandeValeur          = Math.max(SectionAuteurs, Math.max(SectionInformationArticle, SectionBookmarks)); // - 40;
	
	// Fixons la hauteur du div contenant les trois section à la section de hauteur maximale
	classInfoAuth.css("height", plusGrandeValeur+"px");

	/******************************************* Calcul des dimensions + resize de la fenêtre ***************************************/
	calcul_dimensions();

	jQuery(window).resize(
		function() {
			if (classNavSom.hasClass("fixed2")) {
				fixed_desactiver(false);
				largeurSommaire = classNavSom.width();
				fixed_activer(false);
			}

			calcul_dimensions();
			casMobile();
		}
	);

	/******************************************* Si pas IE6 On gère le scroll pour fixer ou non le sommaire *************************/
	if (version_ie() != 6 && hauteurContenuArticle > hauteurSommaire) {
		jQuery(window).scroll(function() {

			// Si la class StopArrondi est présente, alors le menu est déroulé
			if (classToggle.hasClass("StopArrondi")) {

				// Si le scroll est supérieur à la hauteur de l'entête alors on fixe le sommaire
				if (jQuery(window).scrollTop() >= positionSommaire && hauteurSommaire > hauteurDroiteSommaire) {
					fixed_activer(true);
				}
				// Sinon, on retire la class de fixation
				else {
					fixed_desactiver(true);
			  }
			// Si le sommaire est fermé, on supprime la class de fixation
			}
		else {
				fixed_desactiver(true);
		}
		});
	}

	/******************************************* Toggle pour ouvrir et fermer le sommaire ******************************************/
	classToggle.toggle(
		function() {
			classToggle.removeClass("StopArrondi");
			calcul_dimensions();
			jQuery(".nav-sommaire-ul").slideUp("slow",function(){
				TogglePlier.hide();
				ToggleDeplier.show();
				fixed_desactiver(false);
				calcul_dimensions();
			});
		}, function () {
			classToggle.addClass("StopArrondi");
			jQuery(".nav-sommaire-ul").slideDown("slow",function(){
				ToggleDeplier.hide();
				TogglePlier.show();
				calcul_dimensions();
            });
        }
    );
	/************************Mobile et tablette - sommaire *******************/
//alert(navigator.userAgent)
/*
	if( isDeviceMobile() ) {
		//alert("texte");
		// console.log('You are using a mobile device!');
		classToggle.removeClass("StopArrondi");
		calcul_dimensions();
		jQuery(".nav-sommaire-ul").slideUp("slow",function(){
			TogglePlier.hide();
			ToggleDeplier.show();
			fixed_desactiver(false);
			calcul_dimensions();
		});
	}
*/
	/******************************************* Présence de pub sur les faqs *****************************************************/
	presencePubFaq();

	/******************************************* Onglets pour les livres **********************************************************/
	jQuery(".classLivreContenu").addClass("displayNone");
	jQuery(".livreOnglets .classJsInfo").addClass("livreOngletActif");
	jQuery(".livreDetails .classJsInfo").removeClass("displayNone");

	jQuery(".classLivreOnglet").click(
		function() {

			// On récupère la première class
			var parentId = jQuery(this).parent().parent().attr("id");
			var divClass = jQuery(this).attr("class");
			var recupDivClass = divClass.split(" ");
			var classe1 = recupDivClass[0];

			jQuery("#" + parentId + " .classLivreOnglet").removeClass("livreOngletActif");
			jQuery(this).addClass("livreOngletActif");
			jQuery("#" + parentId + " .classLivreContenu").addClass("displayNone");
			jQuery("#" + parentId + " .livreDetails ."+classe1).removeClass("displayNone");
		}
	);

	/******************************************* Gestion des popup des auteurs *****************************************************/
	popup_modal_auteur();

	// Améliorer le sommaire des pages de cours de developpez.com
	if ( document.URL.indexOf("page=") < 0 ) {
		if ( jQuery(".PageCours").length >=4 && jQuery(".ThemeCours").length >= 30 ) {
			jQuery(".PageCours").prepend('<img src="https://www.developpez.com/template/kit/kitplus.png"/>');
			jQuery(".ListeCategorieCours").hide();
		} else {
			jQuery(".PageCours").prepend('<img src="https://www.developpez.com/template/kit/kitmoins.png"/>');
		}
		jQuery(".PageCours > img").css('cursor','pointer');
 
		jQuery(".PageCours > img").mouseover(function() {
			jQuery(this).next("a").css('background', "#DDDDFF");
		});
		jQuery(".PageCours > img").mouseleave(function() {
			jQuery(this).next("a").css('background', "#FFFFFF");
		});
 
		jQuery(".PageCours > img").click(function() {
			if( jQuery(this).attr("src") == "https://www.developpez.com/template/kit/kitplus.png" ) {
				jQuery(this).attr("src", "https://www.developpez.com/template/kit/kitmoins.png" );
				jQuery(this).next('a').next("ul").slideDown();
			} else {
				jQuery(this).attr("src", "https://www.developpez.com/template/kit/kitplus.png" );
				jQuery(this).next('a').next("ul").slideUp();
			}
		});
	}
	/*
	if ( document.URL.indexOf("page=") > 0 ) {
		console.log('ancre : ' + location.hash );
		console.log('search : ' + location.search );
	}
	*/
});

/**********************************************************************************************************************************/
/******************************************* FONCTION POUR VERSION IE *************************************************************/
/**********************************************************************************************************************************/
function version_ie() {
	if (navigator.appVersion.indexOf('MSIE 6') > 0)
		return 6;
	else if (navigator.appVersion.indexOf('MSIE 7') > 0)
		return 7;
	else if (navigator.appVersion.indexOf('MSIE 8') > 0)
		return 8;
	else
		return 0;
}

/**********************************************************************************************************************************/
/******************************************* FONCTION POUR FIXER OU NON LE SOMMAIRE ***********************************************/
/**********************************************************************************************************************************/
function fixed_activer(licence) {
	if ( $( window ).width() < LargeurEcranMobile ) { return; }
	classNavSom.addClass("fixed2");
	if (version_ie() == 7) {
		classFixeMarge.addClass("fixed3");
		classInfoAuth.addClass("fixed3");
	  classNavSom.css("width", (largeurSommaire + 10) + "px");
  }
	else if (version_ie() == 8) {
		classNavSom.css("margin-left", "10px");
		classNavSom.css("width", largeurSommaire + "px");
	}
	else {
		classNavSom.css("width", (largeurSommaire - 10) + "px");
	}
	if (licence == true) {
		jQuery(".licence").css("margin-left", largeurSommaire + "px");
	}
}

function fixed_desactiver(licence) {
	if ( $( window ).width() < LargeurEcranMobile ) { return; }
	classNavSom.removeClass("fixed2");
	if (version_ie() == 7) {
		classFixeMarge.removeClass("fixed3");
		classInfoAuth.removeClass("fixed3");
	}
	else if (version_ie() == 8) {
		classNavSom.css("margin-left", "");
	}
	classNavSom.css("width", "");
	if (licence == true) {
		jQuery(".licence").css("margin-left", "");
	}
}

/**********************************************************************************************************************************/
/******************************************* FONCTION POUR PLACER LE CONTENU ******************************************************/
/**********************************************************************************************************************************/
function contenuEnDessous(contenuDessous) {
	articleBody.css("margin-left", "0px");
	articleBody.css("margin-top", "0px");
	articleBody.css("clear", "both");
	if (contenuDessous == true)
		barreNavigation.css("margin-top", "20px");
}

function contenuSuivie(contenuMT, navMT) {
	articleBody.css("margin-left", largeurSommaire + 15 + "px");
	articleBody.css("margin-top", contenuMT+"px");
	barreNavigation.css("margin-top", navMT+"px");
	articleBody.css("clear", "none");
}

/**********************************************************************************************************************************/
/******************************************* FONCTION POUR CALCULER LES DIMENSIONS SELON L'ECRAN **********************************/
/**********************************************************************************************************************************/
function calcul_dimensions() {
	if ( $( window ).width() < LargeurEcranMobile ) {
		contenuEnDessous(true);
		return;
	}

  // <ul> principal du sommaire
	var classNavSomUlPrinc = jQuery("#nav_sommaire_ul_principal");

	largeurSommaire = classNavSom.width();

	var hautmarge, hautinfo;
	hautmarge = classFixeMarge.height;
	hautinfo = classInfoAuth.height;
	if (hauteurContenuArticle > hauteurSommaire || classNavSomUlPrinc.css("overflow-y") == "scroll") {
		// Hauteur du sommaire du synopsis (son div)
		hauteurDroiteSommaire = classFixeMarge.height();
	}
	else {
		// Hauteur du sommaire du synopsis (son div) + div des 3 sections + un rajout
		hauteurDroiteSommaire = classFixeMarge.height() + classInfoAuth.height() + 200; // 200px d'espace blanc autorisé
	}

	// Si il n'y a ni synopsis ni d'infos
	if (verifPresenceFixeMarge == 0 && verifPresenceInfoAuth == 0) {

		if (hauteurSommaire < 50 && hauteurSommaire < hauteurContenuArticle) {
			contenuEnDessous(false);
		}
		else if (ToggleDeplier.css("display") == "none") {
			contenuSuivie(0, "-25");
		}
		else {
			contenuEnDessous(true);
	  }
	// Sinon, sommaire plus grand que la parti droite | le sommaire est remonté | pas ie6 | article plus grand que sommaire
	} else if (hauteurSommaire > hauteurDroiteSommaire && classToggle.hasClass("StopArrondi") && version_ie() != 6 && hauteurContenuArticle > hauteurSommaire) {
		contenuSuivie(20, 0);
	}
	// Sinon
	else {
		contenuEnDessous(false);
  }
	/*console.log('Hauteur= ' + jQuery(window).height());
	console.log('gabarit_pied= ' + jQuery("#gabarit_pied").height());
	console.log('hauteurSommaire= ' + hauteurSommaire);
	console.log('margeBas= ' + margeBas);*/
	if (jQuery(window).height() - jQuery("#gabarit_pied").height() < hauteurSommaire && classToggle.hasClass("StopArrondi")) {
		classNavSomUlPrinc.css("overflow-y", "scroll").css("height", jQuery(window).height() - margeBas - 50);
		classNavSom.css("height", jQuery(window).height() - margeBas);
		//console.log('sommaire grand');
	} 
	
	else {
		//console.log('sommaire petit');
		classNavSomUlPrinc.css("overflow-y", "auto").css("height", "");
		classNavSom.css("height", "");
	}
}

/**********************************************************************************************************************************/
/******************************************* FONCTION POUR AFFICHER/CACHER UN CODE ************************************************/
/**********************************************************************************************************************************/
function CacherMontreCode(elementId, IdTitreCode){

	var ObjetCode = jQuery("#"+elementId);	// Id de ol du code
	var ObjettitreCode = jQuery("#"+IdTitreCode);	// Id du div contenant le titre d'un code

// <AJOUTE>
	if (ObjetCode.parents('.code_avec_lignes').length) { // Si tableau affichant les numéros de ligne
		ObjetCode = ObjetCode.closest('.code_avec_lignes');
	}
// </AJOUTE>

	if (ObjetCode.css("display") == 'none') {

		ObjetCode.css("display", "block");	// Réafficher le code
		ObjettitreCode.removeClass("radius_titre_des_codes_seul");	// On supprime la classe pour enlever l'arrondi inférieur

	} else {
		ObjetCode.css("display", "none");	// Afficher le code
		ObjettitreCode.addClass("radius_titre_des_codes_seul");	// On rajoute la classe pour modifier l'arrondi inférieur
	}
}


/**********************************************************************************************************************************/
/******************************************* FONCTION POUR AFFICHER/CACHER UNE PARTIE DE SOMMAIRE ************************************************/
/**********************************************************************************************************************************/
function CacherMontreSommaireFAQ(elementId, IdImage1){

	var ObjetCode     = jQuery("#"+elementId);	// Id de ul du code
	var ObjetIdImage1 = jQuery("#"+IdImage1);	  // Id de ul du code

	if (ObjetCode.css("display") == 'none') {
		ObjetCode.css("display", "block");	// Réafficher le block
		ObjetIdImage1.attr('src', cheminImageKit + 'kitmoins.png');
	} else {
		ObjetCode.css("display", "none");	// Afficher le block
		ObjetIdImage1.attr('src', cheminImageKit + 'kitplus.png');
	}
}


/**********************************************************************************************************************************/
/******************************************* FONCTION SELECTION DE CODE PAR CLIC **************************************************/
/**********************************************************************************************************************************/
function selectionCode( id_source){
  var oSelect, oRange, oSrc = document.getElementById( id_source);
  if( window.getSelection) {
    oSelect = window.getSelection();
    oSelect.removeAllRanges();
    oRange = document.createRange();
    oRange.selectNodeContents( oSrc);
    oSelect.addRange( oRange);
  }
  else {
    oRange = document.body.createTextRange();
    oRange.moveToElementText( oSrc);
    oRange.select();
  }
  
  // Affichage du code si ce dernier est caché
	var ObjetCode = jQuery("#"+id_source);	// Id de ol du code
	if (ObjetCode.css("display") == 'none') {
		ObjetCode.css("display", "block");	// Afficher le code
	}

  return false;
}

/**********************************************************************************************************************************/
/******************************************* FONCTION PRESENCE PUB FAQ ************************************************************/
/**********************************************************************************************************************************/
function presencePubFaq() {
	var RubriqueDroite = jQuery("#RubriqueDroite").length;

	if (RubriqueDroite == 0) {
		jQuery(".ArticleComplet").css("margin", "0 20px");
	}
  else {
	/* Exclusion d'IE7 pour le positionnement de la colonne de droite */
	if ( version_ie() != 7 ) {
		jQuery(".ArticleComplet").css("margin-right", "320px");

		/* Page livres uniquement */
		if ( jQuery(window.location.hash)!="" && jQuery(window.location.hash).length==1 ) {
			if ( jQuery("div").hasClass("UnLivreEntier") ) {
				setTimeout(function() { jQuery(window).scrollTop( jQuery(window.location.hash).offset().top ); }, 1000);
			}
		}
	}

		/******************************************* On deboggue les blocks de codes ******************************************************/
		jQuery("pre.code_uniquement").each(function() {
			if (version_ie() == 6)
				jQuery(this).css("width", jQuery(".titre_des_codes").width() + "px").css("padding-bottom", "20px").css("overflow", "auto");
			else
				//jQuery(this).css("width", (jQuery(this).parent().width() - 12) + "px").css("padding-bottom", "20px").css("overflow", "auto");
				// Le titre s'en va avec le scroll si on fixe une taille lorsque la fenêtre du navigateur est petite.
				jQuery(this).css("padding-bottom", "20px").css("overflow", "auto");

		});

	}
}

/**********************************************************************************************************************************/
/******************************************* FONCTION Pop-up Auteurs ************************************************************/
/**********************************************************************************************************************************/
function popup_modal_auteur() {

	// Lorsque l'on clique sur le lien des auteurs
	jQuery(".poplight").click( function() {

		var DivPopupAuteurs = jQuery("#ListeAuteursDVP");
		var LargeurPopup = DivPopupAuteurs.width();
		var HauteurPopup = DivPopupAuteurs.height();

		// Faire apparaitre la pop-up et ajouter le bouton de fermeture
		DivPopupAuteurs.fadeIn().css({
			'width': LargeurPopup
		}).prepend('<a href="#" class="close"><img src="https://www.developpez.com/template/kit/fermeture-icone.png" class="CloturePopup" title="Fermer Pop-pup" alt="fermeture" /></a>');

		// Margin du pop-up pour centrer le pop-up
		// Ajustement avec le CSS de 80px
		var MarginTopPopup  = (HauteurPopup + 80 ) / 2;
		var MarginLeftPopup = (LargeurPopup + 80 ) / 2;

		// Affectation du margin
		DivPopupAuteurs.css({
			'margin-top' : -MarginTopPopup,
			'margin-left' : -MarginLeftPopup
		});

		// Effet fade-in du fond opaque
		jQuery('body').append('<div id="fade"></div>'); //Ajout du fond opaque noir
		// Apparition du fond - .css({'filter' : 'alpha(opacity=80)'}) pour corriger les bogues de IE
		jQuery('#fade').css({'filter' : 'alpha(opacity=80)'}).fadeIn();

		return false;
	});

	// Fermeture de la pop-up et du fond
	// Au clic sur le bouton ou sur le calque...
	jQuery('a.close, #fade').live('click', function() {
		jQuery('#fade , .popup_block').fadeOut(function() {
			jQuery('#fade, a.close').remove();  //...ils disparaissent ensemble
		});
		return false;
	});
}

/* Nombre de vues de la page */
function afficher_nb_vues( nb ) {
	jQuery("#NbrVues").text(nb);
	
	if ( nb > 0 ) { 
		jQuery(".TextNbrVues").css("display", "block"); 
	}
}

/* Avant d'éviter d'avoir un grand espace blanc pour les vidéos MP4 géré par flowplayer, 
   on a mis une largeur quelconque pour centrer le div et ensuite, au clic sur l'image
   on redimensionne avec la vraie taille de la vidéo.
*/
function redimensionner_video (id, largeur, hauteur) {
	var $balise = jQuery('#' + id);
	// Balise a
	$balise.attr('style', "display:block;width:" + largeur + "px;height:" + hauteur + "px;");	
	// Balise parent qui est une div
	hauteur = parseInt(hauteur) + 30; // Heuteur environ d'une ligne pour contenir la titre de la vidéo
	$balise.parent().attr('style', "margin:0 auto;width:" + largeur + "px;height:" + hauteur + "px;");	
}

function SuppTexteTemp (id) {
	jQuery('#' + id).css('display','none');
}

/* Détection des mobiles, tablettes */
function isDeviceMobile(){

	var isMobile = {
		Android: function() {
			return navigator.userAgent.match(/Android/i);// && navigator.userAgent.match(/mobile|Mobile/i);
		},
		BlackBerry: function() {
			return navigator.userAgent.match(/BlackBerry/i);//|| navigator.userAgent.match(/BB10; Touch/);
		},
		iOS: function() {
			return navigator.userAgent.match(/iPhone|iPod/i);
		},
		Opera: function() {
			return navigator.userAgent.match(/Opera Mini/i);
		},
		Windows: function() {
			return navigator.userAgent.match(/IEMobile/i) || navigator.userAgent.match(/webOS/i) ;
		},
		any: function() {
			return (isMobile.Android() || isMobile.BlackBerry() || isMobile.iOS() || isMobile.Opera() || isMobile.Windows());
		}
	};
	return isMobile.any()
}

// éviter de charger MathJax inutilement
if ( window.MathJax ) { 
	MathJax.Ajax.timeout = 20 * 1000;  // 5 seconds rather than 15 seconds timeout for file access
	MathJax.Hub.Register.StartupHook(
		"HTML-CSS Jax Startup",function () {
			MathJax.OutputJax["HTML-CSS"].Font.timeout = 20 * 1000; // 5 second  rather than 5 second font timeout
		}
	);

	window.MathJax.Hub.Config({
		tex2jax: {
		  inlineMath: [ ["kitxmlcodeinlinelatexdvp","finkitxmlcodeinlinelatexdvp"] ],
		  displayMath: [ ["kitxmlcodelatexdvp","finkitxmlcodelatexdvp"] ],
		  processEscapes: true,
		  processRefs: false
		},
		menuSettings: {
			zoom: "Hover",
			mpContext: true,
			mpMouse: true
		},
		"HTML-CSS": { availableFonts: ["TeX"] },
		//imageFont: null,
		webFont: "TeX"
	});
}

	/* Cas spécifique des mobiles */
function casMobile(){
	var LargeurEcran = $( window ).width();
	/* On réduit la largeur des vidéo à la largeur maximale - 50 */
	jQuery("iframe").each(function() { 
		var larg = jQuery(this).attr('width');
		if ( larg && larg > LargeurEcran ) { jQuery(this).attr('width',LargeurEcranVideo); }
		//console.log('iframe = ' + larg + '[' + LargeurEcran + ']');
	});
	jQuery("video").each(function() { 
		var larg = jQuery(this).attr('width');
		if ( larg && larg > LargeurEcran ) { jQuery(this).attr('width',LargeurEcranVideo); }
	});
	jQuery("object").each(function() { 
		var larg = jQuery(this).attr('width');
		if ( larg && larg > LargeurEcran ) { jQuery(this).attr('width',LargeurEcranVideo); }
	});
}



