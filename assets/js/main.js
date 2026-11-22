    (() => {
      const toggle = document.querySelector(".site-header__toggle");
      const nav = document.querySelector(".site-header__nav");
      const navLinks = [...document.querySelectorAll(".site-header__nav a")];

      const closeMenu = () => {
        nav.classList.remove("is-open");
        toggle.classList.remove("is-active");
        toggle.setAttribute("aria-expanded", "false");
        toggle.setAttribute("aria-label", "Open navigation menu");
      };

      toggle.addEventListener("click", () => {
        const isOpen = nav.classList.toggle("is-open");
        toggle.classList.toggle("is-active", isOpen);
        toggle.setAttribute("aria-expanded", String(isOpen));
        toggle.setAttribute("aria-label", isOpen ? "Close navigation menu" : "Open navigation menu");
      });

      navLinks.forEach((link) => link.addEventListener("click", closeMenu));

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeMenu();
      });

      const sectionMap = new Map(
        navLinks.map((link) => [link.getAttribute("href").slice(1), link])
      );

      if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            navLinks.forEach((link) => link.classList.remove("is-active"));
            sectionMap.get(entry.target.id)?.classList.add("is-active");
          });
        }, { rootMargin: "-28% 0px -62% 0px", threshold: 0 });

        sectionMap.forEach((_, id) => {
          const section = document.getElementById(id);
          if (section) observer.observe(section);
        });
      }

      const heroVideo = document.querySelector("[data-hero-video]");
      const heroStages = document.querySelector("[data-hero-stages]");

      if (heroVideo && heroStages) {
        // The timeline of the hero video: [start (s), stage, vertex-voxel grid].
        const heroTimeline = [
          [0, "structure", "64³"],
          [1.0, "vertices", "64³"],
          [2.6, "vertices", "128³"],
          [3.8, "vertices", "256³"],
          [5.0, "vertices", "512³"],
          [6.2, "vertices", "1024³"],
          [7.6, "edges", "1024³"],
          [12.8, "mesh", "1024³"],
          [16.8, "structure", "64³"],
        ];
        const posterTime = 14.5;
        // With motion reduced the video stays paused: a stage is then shown by the frame that completes it.
        const stageStill = { structure: 0, vertices: 7.5, edges: 12.6, mesh: posterTime };
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        const stageItems = [...heroStages.querySelectorAll("[data-stage]")];
        const levelLabel = heroStages.querySelector("[data-hero-level]");
        let currentEntry = null;
        let frameRequest = 0;

        const renderHeroStage = (time) => {
          let entry = heroTimeline[0];
          heroTimeline.forEach((candidate) => {
            if (time >= candidate[0]) entry = candidate;
          });
          if (entry === currentEntry) return;
          currentEntry = entry;
          stageItems.forEach((item) => {
            const isActive = item.dataset.stage === entry[1];
            item.classList.toggle("is-active", isActive);
            if (isActive) item.setAttribute("aria-current", "step");
            else item.removeAttribute("aria-current");
          });
          if (entry[2]) levelLabel.textContent = entry[2];
        };

        // During a seek the tracker already shows the stage that was asked for.
        const syncHeroStage = () => {
          if (!heroVideo.seeking) renderHeroStage(heroVideo.currentTime);
        };

        const trackHeroStage = () => {
          syncHeroStage();
          frameRequest = heroVideo.paused ? 0 : window.requestAnimationFrame(trackHeroStage);
        };

        const seekHero = (time) => {
          const seek = () => {
            heroVideo.currentTime = time;
          };
          if (heroVideo.readyState >= 1) seek();
          else heroVideo.addEventListener("loadedmetadata", seek, { once: true });
        };

        heroVideo.addEventListener("play", () => {
          if (!frameRequest) trackHeroStage();
        });
        heroVideo.addEventListener("seeked", syncHeroStage);

        // A click on a stage jumps to where it starts (a keyframe of the video) and plays on from there.
        stageItems.forEach((item) => {
          item.addEventListener("click", () => {
            const stage = item.dataset.stage;
            const time = reducedMotion ? stageStill[stage] : heroTimeline.find((entry) => entry[1] === stage)[0];
            renderHeroStage(time);
            seekHero(time);
            if (!reducedMotion) heroVideo.play().catch(() => {});
          });
        });

        if (reducedMotion) {
          // No motion: hold the finished mesh, the same frame as the poster.
          heroVideo.autoplay = false;
          heroVideo.pause();
          seekHero(posterTime);
        } else if ("IntersectionObserver" in window) {
          new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) heroVideo.play().catch(() => {});
            else heroVideo.pause();
          }, { threshold: 0.05 }).observe(heroVideo);
        }

        // Autoplay may have started before this script ran, in which case "play" already fired.
        if (!heroVideo.paused) trackHeroStage();
      }

      const formatIndex = (value) => String(value).padStart(2, "0");

      document.querySelectorAll("[data-viewer-group]").forEach((group) => {
        if (group.hasAttribute("data-mesh-viewer-group")) return;

        const setCount = Number(group.dataset.setCount) || 1;
        const slots = [...group.querySelectorAll("[data-viewer-slot]")];
        const counter = group.querySelector("[data-viewer-counter]");
        const status = group.querySelector("[data-viewer-status]");
        const previousButton = group.querySelector("[data-viewer-prev]");
        const nextButton = group.querySelector("[data-viewer-next]");
        const isMultiSlot = slots.length > 1;
        const counterLabel = group.dataset.counterLabel || (isMultiSlot ? "Set" : "Result");
        const itemLabels = (group.dataset.itemLabels || "")
          .split(",")
          .map((label) => label.trim())
          .filter(Boolean);
        let currentSet = Number(group.dataset.currentSet) || 1;

        const renderViewerSet = () => {
          group.dataset.currentSet = String(currentSet);
          counter.textContent = `${counterLabel} ${formatIndex(currentSet)} / ${formatIndex(setCount)}`;

          slots.forEach((slot, slotIndex) => {
            const placeholder = slot.querySelector(".viewer-placeholder");
            const badge = slot.querySelector(".viewer-badge");
            const title = slot.querySelector("[data-viewer-title]");
            const objectIndex = slotIndex + 1;
            const visualState = ((currentSet + slotIndex - 1) % 3) + 1;

            placeholder.dataset.state = String(visualState);

            if (itemLabels[slotIndex]) {
              const vertexBudget = Number(itemLabels[slotIndex]).toLocaleString("en-US");
              badge.textContent = `${vertexBudget} vertices`;
              title.textContent = `${vertexBudget} Vertices`;
              placeholder.setAttribute(
                "aria-label",
                `Placeholder for page ${currentSet}, mesh generated with ${vertexBudget} vertices`
              );
            } else if (isMultiSlot) {
              badge.textContent = `Set ${formatIndex(currentSet)} · OBJ ${formatIndex(objectIndex)}`;
              title.textContent = `${title.dataset.baseTitle} ${formatIndex(objectIndex)}`;
              placeholder.setAttribute(
                "aria-label",
                `Placeholder for generation result set ${currentSet}, OBJ ${objectIndex}`
              );
            } else {
              badge.textContent = `Result ${formatIndex(currentSet)}`;
              title.textContent = `${title.dataset.baseTitle} ${formatIndex(currentSet)}`;
              placeholder.setAttribute(
                "aria-label",
                `Placeholder for ${title.dataset.baseTitle.toLowerCase()} ${currentSet}`
              );
            }
          });

          status.textContent = `${counterLabel} ${currentSet} of ${setCount} displayed.`;
        };

        previousButton.addEventListener("click", () => {
          currentSet = currentSet === 1 ? setCount : currentSet - 1;
          renderViewerSet();
        });

        nextButton.addEventListener("click", () => {
          currentSet = currentSet === setCount ? 1 : currentSet + 1;
          renderViewerSet();
        });

        renderViewerSet();
      });

      const copyButton = document.querySelector("[data-copy-citation]");
      const citation = document.getElementById("bibtex");

      copyButton.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(citation.textContent);
          copyButton.textContent = "Copied";
          window.setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1600);
        } catch {
          const range = document.createRange();
          range.selectNodeContents(citation);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          copyButton.textContent = "Selected";
        }
      });
    })();
