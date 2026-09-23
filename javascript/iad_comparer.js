/* Impact ADetailer before/after comparer.
 *
 * Rules that keep the two sides honest:
 *   LEFT  = original  -> the snapshot the backend took before the Impact loop.
 *   RIGHT = detailed  -> whatever the result gallery is showing right now.
 *
 * The right side is pulled straight from the gallery <img> whenever we can
 * match it to a stored pair, so the "detailed" label always sits over the
 * picture the user actually got. The backend copy is only a fallback.
 *
 * Exactly one comparer exists on the page. It is moved under the gallery of
 * whichever tab is open instead of being cloned per gallery.
 */

(function () {
    "use strict";

    var POLL_MS = 500;
    var DOCK_ID = "iad-dock";
    var pct = 50;
    var lastKey = "";
    var armed = false;
    var minRev = 0;
    var armedRun = -1;
    var dragging = false;
    var reportedBuild = "";

    function ensureStyle() {
        if (document.getElementById("iad-comparer-style")) return;
        var s = document.createElement("style");
        s.id = "iad-comparer-style";
        s.textContent = [
            "#iad-dock { margin: 8px 0 0; max-width: 1100px; }",
            "#iad-dock .iad-title { font-size: 13px; font-weight: 600; opacity: .75;",
            "  margin: 0 0 6px; line-height: 1.3; }",
            ".iad-stage { --iad-pct: 50%; position: relative; overflow: hidden; border-radius: 8px;",
            "  background: #111; cursor: ew-resize; user-select: none; line-height: 0; touch-action: none; }",
            ".iad-stage img { display: block; -webkit-user-drag: none; pointer-events: none; }",
            ".iad-detailed { width: 100%; height: auto; }",
            ".iad-original { position: absolute; left: 0; top: 0; width: 100%; height: 100%;",
            "  object-fit: fill; clip-path: inset(0 calc(100% - var(--iad-pct)) 0 0); z-index: 1; }",
            ".iad-bar { position: absolute; top: 0; bottom: 0; left: var(--iad-pct); width: 3px;",
            "  margin-left: -1px; background: #fff; pointer-events: none; z-index: 2;",
            "  box-shadow: 0 0 0 1px rgba(0,0,0,.35); }",
            ".iad-bar:after { content: ''; position: absolute; top: 50%; left: 50%; width: 26px; height: 26px;",
            "  margin: -13px 0 0 -13px; border: 2px solid #fff; border-radius: 50%; background: rgba(0,0,0,.35); }",
            ".iad-tag { position: absolute; top: 8px; z-index: 3; font-size: 11px; letter-spacing: .04em;",
            "  text-transform: uppercase; color: #fff; background: rgba(0,0,0,.5); padding: 2px 6px;",
            "  border-radius: 3px; pointer-events: none; }",
            ".iad-tag-l { left: 8px; }",
            ".iad-tag-r { right: 8px; }",
            ".iad-slider { width: 100%; margin: 8px 0 0; display: block; }"
        ].join("\n");
        document.head.appendChild(s);
    }

    function visible(el) {
        return !!(el && el.offsetParent !== null && el.getClientRects().length);
    }

    /* ---------- gallery lookup ---------- */

    function activeGallery() {
        var ids = ["txt2img_gallery", "img2img_gallery"];
        for (var i = 0; i < ids.length; i++) {
            var el = document.getElementById(ids[i]);
            if (visible(el)) return el;
        }
        for (var j = 0; j < ids.length; j++) {
            var fb = document.getElementById(ids[j]);
            if (fb) return fb;
        }
        return null;
    }

    /* Map the selected gallery thumbnail onto a stored pair.
       Returns a slot index, or -1 to mean "just use the last pair". */

    /* ---------- dock ---------- */

    function dock() {
        var d = document.getElementById(DOCK_ID);
        if (!d) {
            d = document.createElement("div");
            d.id = DOCK_ID;
            d.style.display = "none";
        }
        var gallery = activeGallery();
        var host = gallery ? (gallery.closest(".image-gallery") || gallery) : null;
        var anchor = host || document.getElementById("txt2img_results") || document.body;
        if (d.previousElementSibling !== anchor && d.parentElement !== anchor) {
            if (host) anchor.insertAdjacentElement("afterend", d);
            else anchor.appendChild(d);
        }
        // Any stray dock from an older build of this extension.
        document.querySelectorAll(".iad-dock, #iad-fallback-dock").forEach(function (old) {
            if (old !== d && old.parentElement) old.parentElement.removeChild(old);
        });
        return d;
    }

    function setPct(v) {
        pct = Math.max(0, Math.min(100, Number(v) || 0));
        var stage = document.querySelector("#" + DOCK_ID + " .iad-stage");
        if (stage) stage.style.setProperty("--iad-pct", pct + "%");
        var sl = document.querySelector("#" + DOCK_ID + " .iad-slider");
        if (sl && sl.value !== String(Math.round(pct))) sl.value = String(Math.round(pct));
    }

    function bind(d) {
        var stage = d.querySelector(".iad-stage");
        if (!stage || stage.dataset.bound === "1") return;
        stage.dataset.bound = "1";

        var move = function (clientX) {
            var r = stage.getBoundingClientRect();
            setPct(((clientX - r.left) / Math.max(1, r.width)) * 100);
        };

        stage.addEventListener("pointerdown", function (ev) {
            ev.preventDefault();
            dragging = true;
            try { stage.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
            move(ev.clientX);
        });
        stage.addEventListener("pointermove", function (ev) {
            if (dragging) move(ev.clientX);
        });
        var stop = function (ev) {
            dragging = false;
            try { stage.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
        };
        stage.addEventListener("pointerup", stop);
        stage.addEventListener("pointercancel", stop);

        var sl = d.querySelector(".iad-slider");
        if (sl) sl.addEventListener("input", function () { setPct(sl.value); });

        setPct(pct);
    }

    function render(d, originalSrc, detailedSrc, hint) {
        d.innerHTML =
            '<div class="iad-title" title="' + hint + '">ADetailer Comparer</div>' +
            '<div class="iad-stage">' +
            '<img class="iad-detailed" alt="detailed" draggable="false" src="' + detailedSrc + '"/>' +
            '<img class="iad-original" alt="original" draggable="false" src="' + originalSrc + '"/>' +
            '<div class="iad-bar"></div>' +
            '<span class="iad-tag iad-tag-l">original</span>' +
            '<span class="iad-tag iad-tag-r">detailed</span>' +
            "</div>" +
            '<input class="iad-slider" type="range" min="0" max="100" value="' + Math.round(pct) + '"/>';
        d.style.display = "";
        bind(d);
    }

    /* ---------- poll ---------- */

    /* Works both at the site root and behind a sub-path reverse proxy.
       The prefix that answers first is reused for the image endpoint. */
    var prefixes = ["", window.location.pathname.replace(/\/[^/]*$/, "").replace(/\/$/, "")];
    var apiPrefix = null;

    function getState() {
        var list = apiPrefix === null ? prefixes : [apiPrefix];
        var i = 0;
        function attempt() {
            if (i >= list.length) { apiPrefix = null; return Promise.resolve(null); }
            var p = list[i++];
            return fetch(p + "/impact_adetailer/state?t=" + Date.now(), { cache: "no-store" })
                .then(function (r) {
                    if (!r.ok) return attempt();
                    apiPrefix = p;
                    return r.json();
                })
                .catch(function () { return attempt(); });
        }
        return attempt();
    }

    function imageUrl(slot, side, run, rev) {
        return (apiPrefix || "") + "/impact_adetailer/image?slot=" + slot +
            "&side=" + side + "&r=" + run + "&v=" + (rev || 0);
    }

 
    function hideDock() {
        armed = true;
        minRev = window._iadRev || 0;
        lastKey = "";
        var node = document.getElementById(DOCK_ID);
        if (!node) return;
        node.style.display = "none";
        node.innerHTML = "";
    }

    function hookGenerate() {
        ["txt2img_generate", "img2img_generate"].forEach(function (id) {
            var b = document.getElementById(id);
            if (!b || b.dataset.iadHook === "1") return;
            b.dataset.iadHook = "1";
            b.addEventListener("click", hideDock, true);
        });
    }

    function tick() {
        var d = document.getElementById(DOCK_ID);
        if (dragging && d) return;
        getState().then(function (data) {
            var node = dock();
            var rev = (data && typeof data.rev === "number") ? data.rev : 0;
            window._iadRev = rev;

            if (armed) {
                if (!data || !data.count || rev <= minRev) {
                    node.style.display = "none";
                    node.innerHTML = "";
                    lastKey = "";
                    return;
                }
                armed = false;
                lastKey = "";
            }

            if (!data || !data.ok || !data.show || !data.count) {
                node.style.display = "none";
                node.innerHTML = "";
                lastKey = "";
                return;
            }

            var rec = data.pairs[data.pairs.length - 1];
            if (!rec) return;

            var originalSrc = imageUrl(rec.slot, "original", data.run, rev);
            var detailedSrc = imageUrl(rec.slot, "detailed", data.run, rev);
            var key = [data.run, rec.slot, rev].join("|");
            if (key === lastKey && node.querySelector(".iad-stage")) {
                bind(node);
                return;
            }
            lastKey = key;
            render(
                node, originalSrc, detailedSrc,
                "#" + rec.index + " · run " + data.run + " · rev " + rev
            );
        });
    }

    function boot() {
        ensureStyle();
        hookGenerate();
        document.addEventListener("keydown", function (ev) {
            if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") hideDock();
        }, true);
        tick();
        setInterval(function () {
            hookGenerate();
            tick();
        }, POLL_MS);
    }

    if (typeof onUiLoaded === "function") onUiLoaded(boot);
    else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
})();