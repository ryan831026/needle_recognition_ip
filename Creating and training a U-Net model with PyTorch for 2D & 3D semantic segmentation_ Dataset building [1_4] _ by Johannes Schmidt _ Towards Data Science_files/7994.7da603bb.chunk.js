(self.webpackChunklite=self.webpackChunklite||[]).push([[7994],{57682:(e,n,i)=>{"use strict";i.d(n,{BG:()=>l,ZD:()=>o,mr:()=>d});var t=i(67294),a=i(77355),l=52,o=48,r={xs:"24px",sm:"24px",md:"".concat(l,"px"),lg:"".concat(l,"px"),xl:"".concat(l,"px")},c={xs:"24px",sm:"24px",md:"".concat(o,"px"),lg:"".concat(o,"px"),xl:"".concat(o,"px")},d=function(e){var n=e.heading,i=e.navbar,l=e.marginBottom,o=e.navbarMargin,d=l?{xs:"24px",sm:"24px",md:"".concat(l,"px"),lg:"".concat(l,"px"),xl:"".concat(l,"px")}:c;return t.createElement(a.x,{marginTop:r,marginBottom:d},t.createElement(a.x,{marginBottom:o||"40px"},n),i)}},81360:(e,n,i)=>{"use strict";i.d(n,{J:()=>v});var t=i(63038),a=i.n(t),l=i(23279),o=i.n(l),r=i(67294),c=i(77355),d=i(14646),s=i(31889),u=i(34135),m=i(97),k=i(87879),p=function(){return{display:"flex",alignItems:"center",overflowX:"scroll",overflowY:"hidden",padding:"2px 0","::-webkit-scrollbar":{width:0,background:"transparent"}}},v=function(e){var n=e.children,i=e.childrenWidth,t=(0,d.I)(),l=(0,r.useState)(!1),c=a()(l,2),s=c[0],m=c[1],k=(0,r.useState)(!1),v=a()(k,2),g=v[0],S=v[1],h=(0,r.useState)(),x=a()(h,2),b=x[0],y=x[1],N=(0,r.useCallback)((function(e){e&&y(e)}),[]),E=(0,r.useCallback)((function(){b&&b.scrollWidth>b.clientWidth&&m(!0)}),[b]);(0,r.useEffect)((function(){return u.V6.on("resize",E),E(),function(){return u.V6.off("resize",E)}}),[E]);var F=(0,r.useCallback)((function(e){b&&b.scrollBy({left:e,behavior:"smooth"})}),[b]),C=(0,r.useCallback)((function(){F(100)}),[F]),w=(0,r.useCallback)((function(){F(-100)}),[F]),T=(0,r.useCallback)(o()((function(){if(b){var e=b.scrollWidth-(b.clientWidth+b.scrollLeft);m(e>0),S(b.scrollLeft>0)}}),100),[b]);return(0,r.useEffect)((function(){T()}),[T]),(0,r.useEffect)((function(){b&&i&&m(i>b.clientWidth)}),[i,null==b?void 0:b.clientWidth]),r.createElement("div",{ref:N,onScroll:T,className:t(p)},n,s&&r.createElement(f,{onClick:C,isRight:!0}),g&&r.createElement(f,{onClick:w}))},g={border:"none",padding:"0",margin:"2px",cursor:"pointer"},f=function(e){var n=e.isRight,i=e.onClick,t=(0,s.F)(),a=(0,d.I)(),l=t.backgroundColor,o=(0,k.bv)(l,0);return r.createElement(c.x,{position:"absolute",right:n?"0":void 0,left:n?void 0:"0",top:"0",bottom:"12px",paddingLeft:n?"42px":void 0,paddingRight:n?void 0:"42px",background:"linear-gradient(".concat(n?90:270,"deg, ").concat(o," 0px, ").concat(l," 50%)"),display:"flex",alignItems:"center"},r.createElement("button",{className:a(g),onClick:i,"aria-label":n?"next sections":"previous sections"},r.createElement(m.Z,{height:"26px",width:"26px",style:n?{transform:"rotate(180deg)"}:void 0,"aria-hidden":"true"})))}},42139:(e,n,i)=>{"use strict";i.d(n,{R:()=>m});var t=i(67294),a=i(77355),l=i(66411),o=i(14646),r=i(31889),c=i(63038),d=i.n(c),s=i(81360),u=i(47172),m=function(e){var n,i,c,m,k,p=e.items,v=e.isSearchPage,g=(0,o.I)(),f=(0,r.F)(),S="scroller-items",h=(n="#".concat(S),i=t.useState(),c=d()(i,2),m=c[0],k=c[1],t.useEffect((function(){var e=new ResizeObserver((function(e){var n,i,t=null===(n=e[0])||void 0===n||null===(i=n.contentRect)||void 0===i?void 0:i.width;t&&k(t)})),i=document.querySelector(n);return i&&e.observe(i),function(){e.disconnect()}}),[n]),{width:m}).width;return t.createElement(a.x,{position:"relative",height:"39px",boxShadow:"inset 0 -1px 0 ".concat(f.baseColor.border.lighter),overflow:"hidden"},t.createElement(s.J,{childrenWidth:h},t.createElement("div",{id:S,className:g({display:"flex"})},p.map((function(e,n){var i=n===p.length-1?"50px":e.marginRight,a=e.text,o=e.onClick,r=e.isActive,c=e.target,d=e.tabIndex,s=e.ariaControls,m=e.href;return t.createElement(l.cW,{source:{index:n},extendSource:!0,key:"page-navigation-tab-".concat(a,"-").concat(n)},t.createElement(u.v,{text:a,href:m,onClick:o,isActive:r,target:c,tabIndex:d,ariaControls:s,marginRight:i,isSearchPage:v}))})))))}},47172:(e,n,i)=>{"use strict";i.d(n,{v:()=>k});var t=i(67294),a=i(5977),l=i(77355),o=i(93310),r=i(87691),c=i(14646),d=i(43487),s=i(75101),u=i(42140),m=function(e){return{":hover":{color:"".concat(e.baseColor.fill.darker," !important")}}},k=function(e){var n=(0,c.I)(),i=(0,s.G)(),k=e.target,p=e.text,v=e.isSearchPage,g=e.marginRight,f=void 0===g?"32px":g,S="href"in e?e.href:void 0,h="onClick"in e?e.onClick:void 0,x="tabIndex"in e?e.tabIndex:void 0,b="ariaControls"in e?e.ariaControls:void 0,y="isActive"in e?e.isActive:i(S||""),N=(0,d.v9)((function(e){return e.navigation.currentLocation})),E=(0,a.TH)(),F=(0,u.dD)(E.search).q,C="".concat(N,"?q=").concat(F),w=v?S===C:y||i(null!=S?S:"");return t.createElement(l.x,{marginRight:f,paddingBottom:"16px",minWidth:"max-content",borderBottom:w?"BASE_DARKER":"BASE_LIGHTER"},t.createElement(o.r,{role:"tab",onClick:h,href:S,target:k,tabIndex:x,"aria-controls":b,className:n({border:"none",cursor:"pointer",padding:"0px"})},t.createElement(r.F,{scale:"M",color:w?"DARKER":"LIGHTER"},t.createElement("span",{className:n(m)},p))))}},40358:(e,n,i)=>{"use strict";i.d(n,{E:()=>f});var t=i(67294),a=i(88641),l=i(25550),o=i(49608),r=i(77355),c=i(5977),d=i(77520),s=i(26350),u=i(47230),m=i(87691),k=i(92661),p="three_column_layout_nav",v=function(e){var n,i=e.loading,a=(0,c.TH)(),l=(0,k.$B)(a.pathname),o=null==l||null===(n=l.route)||void 0===n?void 0:n.name;return t.createElement(r.x,{display:"flex",alignItems:"center"},t.createElement(r.x,{flexGrow:"1",flexShrink:"0",playwrightClassName:"pw-susi-button"},t.createElement(s.R,{pageSource:(0,d.x)(o,"register"),operation:"register",susiEntry:p},t.createElement(u.z,{size:"REGULAR",buttonStyle:"BRAND",width:"100%",loading:i,"aria-label":"sign up"},"Get started"))),!i&&t.createElement(r.x,{marginLeft:"24px",playwrightClassName:"pw-sign-in-button"},t.createElement(m.F,{scale:"M",color:"ACCENT"},t.createElement(s.R,{pageSource:(0,d.x)(o,"login"),operation:"login",susiEntry:p},"Sign In"))))},g=i(10974),f=function(){var e=(0,l.r)(),n=e.viewerId,i=e.loading,c=n&&(0,o.Q)(n),d=(0,a.L)();return!c&&d?null:t.createElement(r.x,null,c?t.createElement(g.N,null):t.createElement(v,{loading:i}))}},64423:(e,n,i)=>{"use strict";i.d(n,{d:()=>ne,m:()=>ee});var t=i(67294),a=i(937),l=i(20113),o=i(14646),r=function(e){var n=e.publisher,i=e.scale,a=void 0===i?"XS":i,r=(0,o.I)();return n.name?t.createElement(l.X6,{playwrightClassName:"pw-author-name",scale:a},t.createElement("span",{className:r({wordBreak:"break-word"})},n.name)):null},c=i(41987),d=i(63038),s=i.n(d),u=i(38460),m=i(25468),k=i(65590),p=i(319),v=i.n(p),g=i(13085),f=i(14337),S=i(54341),h=i(84683),x=i(19308),b={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"CollectionTooltip_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"description"}},{kind:"Field",name:{kind:"Name",value:"subscriberCount"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionAvatar_collection"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionFollowButton_collection"}}]}}].concat(v()(h.d.definitions),v()(x.I.definitions))},y={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"UserFollowsListItem_followedEntity"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"FollowedEntity"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherAvatar_publisher"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"hasDomain"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserMentionTooltip_user"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionTooltip_collection"}}]}}]}}].concat(v()(f.v.definitions),v()(S.O.definitions),v()(b.definitions))},N=([{kind:"FragmentDefinition",name:{kind:"Name",value:"UserFollowsList_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"customStyleSheet"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"blogroll"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"visibility"}}]}}]}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"username"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherFollowingCount_publisher"}}]}}].concat(v()(g.b.definitions)),{kind:"Document",definitions:[{kind:"OperationDefinition",operation:"query",name:{kind:"Name",value:"UserFollowsListQuery"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"userId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"ID"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"limit"}},type:{kind:"NamedType",name:{kind:"Name",value:"Int"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"userFollows"},arguments:[{kind:"Argument",name:{kind:"Name",value:"userId"},value:{kind:"Variable",name:{kind:"Name",value:"userId"}}},{kind:"Argument",name:{kind:"Name",value:"limit"},value:{kind:"Variable",name:{kind:"Name",value:"limit"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserFollowsListItem_followedEntity"}}]}}]}}].concat(v()(y.definitions))}),E=i(78757),F=i(30826),C=i(75210),w=i(28695),T=i(77355),I=i(93310),_=i(73917),D=i(42130),R=i(68894),A=function(e){return{"& path":{fill:e.baseColor.text.lighter},":hover path":{fill:e.baseColor.fill.darker},":focus path":{fill:e.baseColor.fill.darker},":hover":{background:e.baseColor.background.normal},padding:"4px",borderRadius:"6px"}},B=function(e){var n=e.popoverRenderFn,i=e.ariaId,a=e.role,l=(0,R.O)(!1),o=s()(l,4),r=o[0],c=o[2],d=o[3];return t.createElement(_.J,{ariaId:i,isVisible:r,placement:"top",targetDistance:10,role:a,popoverRenderFn:n,hide:c,hideOnOutsideClick:!0,noPortal:!0},t.createElement(I.r,{onClick:d,rules:A,"aria-controls":i,"aria-expanded":r},t.createElement(D.Z,null)))},L=i(87691),V=function(e){return{"& p":{wordBreak:"break-all"},":hover p":{textDecoration:"underline",color:e.baseColor.text.light}}},P=function(e){var n=e.entity,i=(0,E.P)(n),a=(0,t.useCallback)((function(){return"User"===n.__typename?t.createElement(w.K,{user:n}):t.createElement(C.L,{collection:n,buttonSize:"COMPACT",buttonStyleFn:function(e){return e?"OBVIOUS":"STRONG"}})}),[n]);return t.createElement(T.x,{tag:"li",display:"flex",justifyContent:"space-between",alignItems:"center"},t.createElement(I.r,{href:i,rules:V},t.createElement(T.x,{display:"flex",paddingRight:"10px"},t.createElement(T.x,{paddingRight:"12px"},t.createElement(F.G,{publisher:n,scale:"XXXS"})),t.createElement(L.F,{clamp:1,scale:"S",color:"LIGHTER"},n.name))),t.createElement(B,{popoverRenderFn:a,ariaId:"creatorInfoPopover-".concat(n.id),role:"dialog"}))},U=i(6443),M=i(18627),H=i(66411),G=i(18122),O=i(97217),X=i(21372),W=function(e){var n,i=e.user,a=i.id,o=i.customStyleSheet,r=(0,U.H)().value,c=a===(null==r?void 0:r.id),d=(0,M.Av)(),p=(0,G.g)({onPresentedFn:function(){return d.event("sidebar.blogrollViewed",{viewerIsAuthor:c})}}),v=(0,u.t)(N,{ssr:!0,variables:{userId:a,limit:5}}),g=s()(v,2),f=g[0],S=g[1],h=S.called,x=S.loading,b=S.error,y=S.data,E=(y=void 0===y?{userFollows:void 0}:y).userFollows,F=(0,m.g)(i),C=F.isFollowingCountVisible,w=F.followingCount,_=(0,t.useState)(),D=s()(_,2),R=D[0],A=D[1],B=(0,t.useState)(!1),V=s()(B,2),W=V[0],z=V[1],Y=(0,t.useCallback)((function(){return z(!0)}),[]),q=(0,t.useCallback)((function(){return z(!1)}),[]);if((null==o||null===(n=o.blogroll)||void 0===n?void 0:n.visibility)===O.n$.BLOGROLL_VISIBILITY_HIDDEN)return null;if(!h)return f(),null;if(x||b||!E||!E.length||w<5)return null;var j=C?"See all (".concat((0,X.rR)(w),")"):"See all";return t.createElement(H.cW,{extendSource:!0,source:{name:"blogrolls_sidebar"}},t.createElement(T.x,{ref:p,position:"relative"},t.createElement(l.X6,{scale:"XS",tag:"span"},"Following"),t.createElement(T.x,{marginTop:"16px",marginBottom:"16px",tag:"ul"},E.map((function(e){return t.createElement(P,{key:null==e?void 0:e.id,entity:e,isTooltipActive:R===e.id,onMouseEnter:function(){return A(e.id)}})}))),t.createElement(L.F,{scale:"S"},t.createElement(I.r,{linkStyle:"SUBTLE",onClick:Y},j)),t.createElement(k.A,{hide:q,publisher:i,followingCount:w,isVisible:W})))},z=i(84739),Y=i(32223),q=i(5422),j=i(25735),J=i(14818),K=i(24330),Q=i(92661),Z=i(87498),$=i(78870),ee=88,ne=function(e){var n=e.includeBlogRoll,i=e.user,l=(0,U.H)().value,o=l&&l.id===i.id,d=(0,z.B)(i),s=(0,j.VB)({name:"enable_lite_user_settings",placeholder:!1}),u=(0,Q.H2)(),m=u("ShowSettings",{}),k=(0,$.Rk)(u("ShowSettingsTab",{setting:Y.B.Account}),{},q.e),p=s?k:m;return t.createElement(t.Fragment,null,t.createElement(I.r,{href:d},t.createElement(J.z,{alt:i.name||"",miroId:i.imageId||Z.gG,diameter:ee})),t.createElement(T.x,{marginTop:"16px"}),t.createElement(I.r,{href:d},t.createElement(r,{publisher:i})),t.createElement(T.x,{marginTop:"4px"}),t.createElement(c.e,{publisher:i}),t.createElement(T.x,{marginTop:"12px"}),t.createElement(a.y,{publisher:i}),o&&t.createElement(T.x,{marginTop:"24px",marginBottom:"46px"},t.createElement(L.F,{scale:"S",color:"ACCENT"},t.createElement(I.r,{href:p},"Edit profile"))),t.createElement(T.x,{marginTop:"24px"}),!o&&t.createElement(H.cW,{source:{name:"publisher_header_actions"},extendSource:!0},t.createElement(T.x,{display:"flex",marginBottom:"40px"},t.createElement(K.N,{creator:i,followButtonSize:"REGULAR",susiEntry:"follow_profile",showMembershipUpsellModal:!0,width:"auto"}))),n&&t.createElement(W,{user:i}))}},1279:(e,n,i)=>{"use strict";i.d(n,{m:()=>t});var t={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherDescription_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"description"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"bio"}}]}}]}}]}},937:(e,n,i)=>{"use strict";i.d(n,{y:()=>r});var t=i(67294),a=i(14294),l=i(38882),o=i(87691),r=function(e){var n=e.publisher,i="Collection"===n.__typename?n.description:n.bio;return i?t.createElement(o.F,{scale:"M"},t.createElement(l.c.Provider,{value:!0},t.createElement(a.P,{wrapLinks:!0},i))):null}},13085:(e,n,i)=>{"use strict";i.d(n,{b:()=>t});var t={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherFollowingCount_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"socialStats"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"followingCount"}},{kind:"Field",name:{kind:"Name",value:"collectionFollowingCount"}}]}},{kind:"Field",name:{kind:"Name",value:"username"}}]}}]}}]}},25468:(e,n,i)=>{"use strict";i.d(n,{g:()=>d,D:()=>s});var t=i(63038),a=i.n(t),l=i(67294),o=i(65590),r=i(93310),c=i(21372),d=function(e){var n,i,t,a,l=0;return"User"===e.__typename&&(l=(null!==(n=null===(i=e.socialStats)||void 0===i?void 0:i.followingCount)&&void 0!==n?n:0)+(null!==(t=null===(a=e.socialStats)||void 0===a?void 0:a.collectionFollowingCount)&&void 0!==t?t:0)),{followingCount:l,isFollowingCountVisible:l>0}},s=function(e){var n=e.publisher,i=e.linkStyle,t=void 0===i?"SUBTLE":i,s=d(n),u=s.followingCount,m=s.isFollowingCountVisible,k=(0,l.useState)(!1),p=a()(k,2),v=p[0],g=p[1];if(!m)return null;var f="".concat((0,c.pY)(u)," Following");return l.createElement(l.Fragment,null,l.createElement(r.r,{linkStyle:t,onClick:function(){return g(!0)}},f),l.createElement(o.A,{hide:function(){return g(!1)},publisher:n,followingCount:u,isVisible:v}))}},65590:(e,n,i)=>{"use strict";i.d(n,{A:()=>p});var t=i(6479),a=i.n(t),l=i(64718),o=i(67294),r=i(19262),c=i(319),d=i.n(c),s=i(69387),u=i(84683),m={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"DialogCollectionEntity_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"slug"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionAvatar_collection"}}]}}].concat(d()(u.d.definitions))},k={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"query",name:{kind:"Name",value:"PublisherFollowingDialogUserQuery"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"id"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"ID"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"pagingInfo"}},type:{kind:"NamedType",name:{kind:"Name",value:"PagingOptions"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"user"},arguments:[{kind:"Argument",name:{kind:"Name",value:"id"},value:{kind:"Variable",name:{kind:"Name",value:"id"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"followingEntityConnection"},arguments:[{kind:"Argument",name:{kind:"Name",value:"paging"},value:{kind:"Variable",name:{kind:"Name",value:"pagingInfo"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"entities"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"DialogUserEntity_user"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"DialogCollectionEntity_collection"}}]}}]}},{kind:"Field",name:{kind:"Name",value:"pagingInfo"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"next"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"to"}},{kind:"Field",name:{kind:"Name",value:"limit"}},{kind:"Field",name:{kind:"Name",value:"from"}}]}}]}}]}}]}}]}}].concat(d()(s.U.definitions),d()(m.definitions))},p=function(e){var n,i,t,c,d,s=e.publisher,u=e.followingCount,m=e.isVisible,p=e.hide,v=(0,l.a)(k,{variables:{id:s.id,pagingInfo:{limit:10}},ssr:!1}),g=v.data,f=v.error,S=v.fetchMore,h=null==g||null===(n=g.user)||void 0===n||null===(i=n.followingEntityConnection)||void 0===i?void 0:i.entities,x=null==g||null===(t=g.user)||void 0===t||null===(c=t.followingEntityConnection)||void 0===c||null===(d=c.pagingInfo)||void 0===d?void 0:d.next;return f||0===u?null:o.createElement(r.L,{hide:p,isVisible:m,heading:"".concat(u," Following"),entities:h,pagingInfoNext:x,handleFetchMore:function(){if(x){x.__typename;var e=a()(x,["__typename"]);S({variables:{pagingInfo:e}})}}})}},32064:(e,n,i)=>{"use strict";i.d(n,{y:()=>l});var t=i(67294),a=i(77355),l=function(e){var n=e.children,i=e.marginTop,l=void 0===i?"40px":i,o=e.paddingBottom,r=void 0===o?"0px":o,c=e.showBorderBottom,d=void 0!==c&&c;return t.createElement(a.x,{marginTop:l,paddingBottom:r,borderBottom:d?"BASE_LIGHTER":"NONE"},n)}},78757:(e,n,i)=>{"use strict";i.d(n,{P:()=>l});var t=i(68427),a=i(84739),l=function(e){var n=(0,t.B)(),i=(0,a.I)();return"Collection"===e.__typename?n(e):"User"===e.__typename?i(e):""}},986:(e,n,i)=>{"use strict";i.d(n,{c:()=>s});var t=i(67294),a=i(88641),l=i(52439),o=i(89636),r=i(76701),c=i(77355),d=i(35010),s=function(e){var n=e.children,i=(0,t.useRef)(0),s=(0,t.useRef)(null),u=(0,t.useRef)("stickyToTop"),m=(0,a.L)(),k=(0,t.useRef)(null),p=(0,o.v)(),v=p.fullNavbarHeight,g=p.addHeightChangeHandler,f=p.removeHeightChangeHandler;return(0,d.L)((function(){var e=function(e){var n=e.currentHeight;s.current&&k.current&&("notSticky"!==u.current&&(s.current.style.top="".concat(n,"px")),k.current.style.minHeight="calc(100vh - ".concat(n,"px)"))};return g(e),function(){f(e)}}),[]),(0,d.L)((function(){var e=function(){if(s.current){var e=window.scrollY>i.current;i.current=window.scrollY;var n=window.innerHeight,t=s.current.offsetHeight-n,a=m?r.f:0;t<=20||(e?("notSticky"===u.current&&window.scrollY>=s.current.offsetTop+t+a&&(u.current="stickyToBottom",s.current.style.position="sticky",s.current.style.top="".concat(-t,"px")),"stickyToTop"===u.current&&(u.current="notSticky",s.current.style.position="relative",s.current.style.top="0px",s.current.style.marginTop="0px",s.current.style.marginTop="".concat(Math.max(i.current-s.current.offsetTop-a,0),"px"))):("notSticky"===u.current&&window.scrollY<=s.current.offsetTop&&(u.current="stickyToTop",s.current.style.position="sticky",s.current.style.top="".concat(a,"px"),s.current.style.marginTop="0px"),"stickyToBottom"===u.current&&(u.current="notSticky",s.current.style.position="relative",s.current.style.top="0px",s.current.style.marginTop="0px",s.current.style.marginTop="".concat(i.current-t-s.current.offsetTop-a,"px"))))}};return window.addEventListener("scroll",e),function(){window.removeEventListener("scroll",e)}})),t.createElement(c.x,{position:"sticky",top:"".concat(v,"px"),ref:s},t.createElement(c.x,{display:"flex",flexDirection:"column",minHeight:"calc(100vh - ".concat(v,"px)"),ref:k},t.createElement(c.x,{flexGrow:"1"},n),t.createElement(l.q,{detailScale:"XS",spacing:"XS"})))}},32223:(e,n,i)=>{"use strict";i.d(n,{B:()=>t,N:()=>p});var t,a=i(67294),l=i(70405),o=i(57682),r=i(42139),c=i(35225),d=i(96370),s=i(77355),u=i(92661),m=i(43487),k=i(71341);!function(e){e.Account="account",e.Publishing="publishing",e.Notifications="notifications",e.Membership="membership",e.Security="security"}(t||(t={}));var p=function(e){var n=e.children,i=e.activeTab,p=(0,k.h)(),v=(0,u.di)("ShowSettings",{}),g=(0,u.di)("ShowSettingsTab",{setting:t.Publishing}),f=(0,u.di)("ShowSettingsTab",{setting:t.Notifications}),S=(0,u.di)("ShowSettingsTab",{setting:t.Membership}),h=(0,u.di)("ShowSettingsTab",{setting:t.Security}),x=(0,a.useMemo)((function(){return[{text:"Account",onClick:function(){return p(v)},isActive:i===t.Account},{text:"Publishing",onClick:function(){return p(g)},isActive:i===t.Publishing},{text:"Notifications",onClick:function(){return p(f)},isActive:i===t.Notifications},{text:"Membership and payment",onClick:function(){return p(S)},isActive:i===t.Membership},{text:"Security and apps",onClick:function(){return p(h)},isActive:i===t.Security}]}),[i,p]),b=(0,m.v9)((function(e){return e.config.productName})),y=(0,a.useMemo)((function(){return a.createElement(c.V,null,"Settings")}),[]);return a.createElement(d.P,{size:"app"},a.createElement(l.ql,null,a.createElement("title",null,"Settings",b?" - ".concat(b):null)),a.createElement(s.x,{paddingBottom:"32px"},a.createElement(o.mr,{heading:y,navbarMargin:{xs:"32px",sm:"32px",md:"56px",lg:"56px",xl:"56px"},navbar:a.createElement(r.R,{items:x})}),n))}},5422:(e,n,i)=>{"use strict";i.d(n,{e:()=>t});var t="#profileInformation"},35225:(e,n,i)=>{"use strict";i.d(n,{V:()=>r});var t=i(67294),a=i(77355),l=i(20113),o={xs:"M",sm:"M",md:"XL",lg:"XL",xl:"XL"},r=function(e){var n=e.children;return t.createElement(a.x,null,t.createElement(l.X6,{scale:o,tag:"h1",fontWeight:"EDITORIAL",clamp:1},n))}},14337:(e,n,i)=>{"use strict";i.d(n,{v:()=>r});var t=i(319),a=i.n(t),l=i(84683),o=i(27048),r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherAvatar_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionAvatar_collection"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserAvatar_user"}}]}}]}}].concat(a()(l.d.definitions),a()(o.W.definitions))}},30826:(e,n,i)=>{"use strict";i.d(n,{G:()=>o});var t=i(67294),a=i(71652),l=i(17193),o=function(e){var n=e.link,i=void 0!==n&&n,o=e.scale,r=void 0===o?"M":o,c=e.publisher;switch(c.__typename){case"User":return t.createElement(l.Yt,{link:i,scale:r,user:c});case"Collection":return t.createElement(a.v,{link:i,size:l.wC[r],collection:c});default:return null}}},75210:(e,n,i)=>{"use strict";i.d(n,{L:()=>d});var t=i(67294),a=i(71652),l=i(82405),o=i(77355),r=i(20113),c=i(87691),d=function(e){var n=e.collection,i=e.buttonSize,d=e.buttonStyleFn,s=n.name,u=n.description;return t.createElement(o.x,{padding:"15px",display:"flex",flexDirection:"column",width:"300px"},t.createElement(o.x,{display:"flex",flexDirection:"row",justifyContent:"space-between",whiteSpace:"normal",borderBottom:"BASE_LIGHTER",paddingBottom:"10px",marginBottom:"10px"},t.createElement(o.x,{display:"flex",flexDirection:"column",paddingRight:"5px"},t.createElement(r.X6,{scale:"S"},s),t.createElement(c.F,{scale:"S"},u)),t.createElement(o.x,null,t.createElement(a.v,{collection:n,link:!0}))),t.createElement(o.x,{display:"flex",flexDirection:"row",alignItems:"center",justifyContent:"space-between"},t.createElement(c.F,{scale:"M"},"Followed by ",n.subscriberCount," people"),t.createElement(l.Fp,{collection:n,buttonSize:i,buttonStyleFn:d,susiEntry:"follow_card"})))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/7994.7da603bb.chunk.js.map