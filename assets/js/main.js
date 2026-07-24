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
