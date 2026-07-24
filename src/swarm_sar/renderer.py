import pygame
import colorsys
from swarm_sar.drone import DroneState


class Renderer:
    """Pygame-based visualizer for the swarm SAR simulation."""

    def __init__(self, simulator):
        """Initializes the Pygame display window and color/font resources.

        Args:
            simulator: The Simulator instance whose state will be rendered.
        """
        self.sim = simulator
        self.show_paths = True
        pygame.init()
        
        env = simulator.env
        # Dynamically scale cell size so that the simulation view is appropriately sized (max 700px)
        max_grid_px = 700
        self.cell_px = max(4, min(16, max_grid_px // max(env.width, env.height)))
        
        grid_w = env.width * self.cell_px
        grid_h = env.height * self.cell_px
        
        self.margin = 25
        self.sidebar_width = 300
        
        # Screen dimensions
        screen_w = grid_w + 2 * self.margin + self.sidebar_width
        screen_h = max(grid_h + 2 * self.margin, 500)
        
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Swarm SAR Coordination")
        
        # Initialize modern fonts
        pygame.font.init()
        try:
            self.font_title = pygame.font.SysFont("Segoe UI, Arial, Helvetica, sans-serif", 16, bold=True)
            self.font_section = pygame.font.SysFont("Segoe UI, Arial, Helvetica, sans-serif", 13, bold=True)
            self.font_body = pygame.font.SysFont("Segoe UI, Arial, Helvetica, sans-serif", 12)
            self.font_bold = pygame.font.SysFont("Segoe UI, Arial, Helvetica, sans-serif", 12, bold=True)
            self.font_mono = pygame.font.SysFont("Consolas, Courier New, monospace", 11)
        except Exception:
            self.font_title = pygame.font.SysFont("arial", 16, bold=True)
            self.font_section = pygame.font.SysFont("arial", 13, bold=True)
            self.font_body = pygame.font.SysFont("arial", 12)
            self.font_bold = pygame.font.SysFont("arial", 12, bold=True)
            self.font_mono = pygame.font.SysFont("monospace", 11)

        # Color system
        self.colors = {
            "bg": (15, 23, 42),            # Deep slate blue
            "grid_bg": (30, 41, 59),       # Dark slate grid box
            "empty": (21, 29, 43),         # Empty cell dark bg
            "searched": (52, 211, 153),    # Glowing emerald/mint green
            "obstacle": (71, 85, 105),     # Slate grey obstacles
            "home": (245, 158, 11),        # Amber home base
            "dead": (239, 68, 68),         # Coral red for crashed drones
            "panel": (30, 41, 59),         # Slate-800 panel
            "panel_dark": (15, 23, 42),    # Slate-900 nested panel
            "panel_border": (51, 65, 85),  # Slate-700 border
            "text": (248, 250, 252),       # Crisp off-white
            "text_muted": (148, 163, 184), # Slate muted text
        }

        # Drone state specific colors
        self.state_colors = {
            "IDLE": (99, 102, 241),        # Indigo
            "SEARCHING": (16, 185, 129),   # Emerald
            "RETURNING": (245, 158, 11),   # Amber
            "REPORTING": (168, 85, 247),   # Purple
            "DEAD": (239, 68, 68),         # Coral
        }

    def get_drone_color(self, drone_id: int) -> tuple[int, int, int]:
        """Returns a unique color for a drone ID using golden-angle HSV distribution.

        Args:
            drone_id: The drone's unique identifier.

        Returns:
            An (R, G, B) tuple with values in [0, 255].
        """
        h = (drone_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.6, 0.85)
        return (int(r * 255), int(g * 255), int(b * 255))

    def get_trail_color(self, drone_id: int) -> tuple[int, int, int]:
        """Returns a dimmed version of the drone color for searched-cell trails.

        Args:
            drone_id: The drone's unique identifier.

        Returns:
            An (R, G, B) tuple blended toward the empty-cell background color.
        """
        drone_color = self.get_drone_color(drone_id)
        # Blend 30% drone color with 70% empty cell background (21, 29, 43) to make a nice dark pastel shade
        return (
            int(drone_color[0] * 0.30 + 21 * 0.70),
            int(drone_color[1] * 0.30 + 29 * 0.70),
            int(drone_color[2] * 0.30 + 43 * 0.70)
        )

    def draw(self):
        """Renders the full simulation state: grid, drones, paths, target, and sidebar HUD."""
        env = self.sim.env
        mm = self.sim.mm
        screen = self.screen
        grid_w = env.width * self.cell_px
        grid_h = env.height * self.cell_px

        # 1. Fill base background
        screen.fill(self.colors["bg"])

        # 2. Draw grid background container panel
        grid_panel = pygame.Rect(
            self.margin - 4,
            self.margin - 4,
            grid_w + 8,
            grid_h + 8
        )
        pygame.draw.rect(screen, self.colors["panel_border"], grid_panel, border_radius=4)
        pygame.draw.rect(screen, self.colors["grid_bg"], pygame.Rect(self.margin, self.margin, grid_w, grid_h))

        # 3. Draw grid cells
        for y in range(env.height):
            for x in range(env.width):
                cell_rect = pygame.Rect(
                    self.margin + x * self.cell_px,
                    self.margin + y * self.cell_px,
                    self.cell_px,
                    self.cell_px,
                )

                if env.grid[y, x] == 2:
                    color = self.colors["home"]
                elif env.grid[y, x] == 1:
                    color = self.colors["obstacle"]
                elif (x, y) in mm.searched:
                    drone_id = mm.searched_by.get((x, y), 0)
                    color = self.get_trail_color(drone_id)
                else:
                    color = self.colors["empty"]

                if self.cell_px >= 6:
                    inset_rect = pygame.Rect(
                        cell_rect.x + 1,
                        cell_rect.y + 1,
                        cell_rect.width - 1,
                        cell_rect.height - 1
                    )
                    pygame.draw.rect(screen, color, inset_rect)
                else:
                    pygame.draw.rect(screen, color, cell_rect)

        # 4. Draw planned flight paths
        if self.show_paths:
            for d in self.sim.drones:
                if not d.alive or not d.path:
                    continue
                points = []
                # Start path lines at drone's current exact pixel coordinates
                points.append((
                    self.margin + d.pos[0] * self.cell_px + self.cell_px // 2,
                    self.margin + d.pos[1] * self.cell_px + self.cell_px // 2
                ))
                for px, py in d.path:
                    points.append((
                        self.margin + px * self.cell_px + self.cell_px // 2,
                        self.margin + py * self.cell_px + self.cell_px // 2
                    ))
                if len(points) >= 2:
                    path_color = self.get_drone_color(d.id)
                    pygame.draw.lines(screen, path_color, False, points, 1)

        # 5. Draw drones as circular markers
        for d in self.sim.drones:
            cx = self.margin + d.pos[0] * self.cell_px + self.cell_px // 2
            cy = self.margin + d.pos[1] * self.cell_px + self.cell_px // 2
            r = max(self.cell_px // 2 + 1, 3)
            
            # Fill color based on status
            color = self.get_drone_color(d.id) if d.alive else self.colors["dead"]
            pygame.draw.circle(screen, color, (cx, cy), r)
            
            # Small black inner point or outline for aesthetics
            pygame.draw.circle(screen, (0, 0, 0), (cx, cy), r, 1)

        # 5b. Draw target if it exists
        if env.target:
            tx, ty = env.target
            cx = self.margin + tx * self.cell_px + self.cell_px // 2
            cy = self.margin + ty * self.cell_px + self.cell_px // 2
            
            target_found = env.target in mm.searched
            if target_found:
                # Draw a bright glowing target mark (solid green/gold star/circle)
                pygame.draw.circle(screen, (245, 158, 11), (cx, cy), max(self.cell_px // 2 + 3, 5))
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), max(self.cell_px // 2 + 1, 3))
                pygame.draw.circle(screen, (245, 158, 11), (cx, cy), 1)
            else:
                # Draw a faint target mark (hollow red crosshair) for operator visibility
                pygame.draw.circle(screen, (239, 68, 68), (cx, cy), max(self.cell_px // 2 + 2, 4), 1)
                pygame.draw.line(screen, (239, 68, 68), (cx - 3, cy), (cx + 3, cy), 1)
                pygame.draw.line(screen, (239, 68, 68), (cx, cy - 3), (cx, cy + 3), 1)

        # 6. Render Sidebar Dashboard
        sb_x = grid_w + self.margin * 2 + 10
        sb_y = self.margin
        sb_w = self.sidebar_width - 20

        # Draw sidebar backdrop
        sb_panel_rect = pygame.Rect(sb_x - 15, 0, self.sidebar_width + 15, screen.get_height())
        pygame.draw.rect(screen, self.colors["panel"], sb_panel_rect)
        pygame.draw.line(screen, self.colors["panel_border"], (sb_x - 15, 0), (sb_x - 15, screen.get_height()), 2)

        # Header Title
        title_lbl = self.font_title.render("SWARM SEARCH & RESCUE", True, self.colors["text"])
        screen.blit(title_lbl, (sb_x, sb_y))
        pygame.draw.line(screen, self.colors["panel_border"], (sb_x, sb_y + 25), (sb_x + sb_w, sb_y + 25), 1)

        # Simulation stats card
        card_y = sb_y + 40
        card_h = 100
        card_rect = pygame.Rect(sb_x, card_y, sb_w, card_h)
        pygame.draw.rect(screen, self.colors["panel_dark"], card_rect, border_radius=4)
        pygame.draw.rect(screen, self.colors["panel_border"], card_rect, 1, border_radius=4)

        active = sum(1 for d in self.sim.drones if d.alive)
        crashed = len(mm.failure_log)
        
        # Display Text Stats
        tick_txt = self.font_body.render(f"Tick: {self.sim.tick}", True, self.colors["text"])
        drones_txt = self.font_body.render(f"Drones Active: {active} / {len(self.sim.drones)}", True, self.colors["text"])
        crashed_txt = self.font_body.render(f"Crashed: {crashed}", True, self.colors["text"] if crashed == 0 else self.colors["dead"])
        
        target_found = env.target is not None and env.target in mm.searched
        target_status = "FOUND" if target_found else "SEARCHING"
        target_col = self.colors["searched"] if target_found else self.colors["text_muted"]
        target_txt = self.font_bold.render(f"Target: {target_status}", True, target_col)
        
        screen.blit(tick_txt, (sb_x + 12, card_y + 12))
        screen.blit(drones_txt, (sb_x + 12, card_y + 32))
        screen.blit(crashed_txt, (sb_x + 12, card_y + 52))
        screen.blit(target_txt, (sb_x + 150, card_y + 12))

        # Coverage Progress Bar
        cov_val = mm.coverage()
        cov_pct_str = f"{cov_val * 100:.1f}%"
        
        cov_lbl = self.font_bold.render(f"Coverage: {cov_pct_str}", True, self.colors["text"])
        screen.blit(cov_lbl, (sb_x + 12, card_y + 75))

        bar_x = sb_x + 120
        bar_y = card_y + 78
        bar_w = sb_w - 132
        bar_h = 10
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(screen, self.colors["searched"], (bar_x, bar_y, int(bar_w * cov_val), bar_h), border_radius=3)

        # 7. Drone Status List Header
        list_y = card_y + card_h + 20
        list_hdr = self.font_section.render("ACTIVE DRONES & BATTERY", True, self.colors["text_muted"])
        screen.blit(list_hdr, (sb_x, list_y))
        pygame.draw.line(screen, self.colors["panel_border"], (sb_x, list_y + 18), (sb_x + sb_w, list_y + 18), 1)

        # List items
        item_y = list_y + 25
        row_h = 24
        
        for d in self.sim.drones:
            # Check off-screen limits for long drone lists
            if item_y + row_h > screen.get_height() - 15:
                break
                
            # Draw colored indicator dot next to drone ID
            dot_color = self.get_drone_color(d.id) if d.alive else self.colors["dead"]
            pygame.draw.circle(screen, dot_color, (sb_x + 5, item_y + 8), 4)
            
            # Drone ID and Coordinates
            d_pos_str = f"[{d.pos[0]:02d},{d.pos[1]:02d}]"
            d_info = f"D{d.id:02d} {d_pos_str}"
            info_lbl = self.font_mono.render(d_info, True, self.colors["text"])
            screen.blit(info_lbl, (sb_x + 14, item_y + 2))

            # State tag background and label
            state_str = d.state.name if d.alive else "DEAD"
            state_col = self.state_colors.get(state_str, (100, 100, 100))
            
            tag_rect = pygame.Rect(sb_x + 95, item_y + 1, 65, 17)
            pygame.draw.rect(screen, state_col, tag_rect, border_radius=3)
            
            tag_lbl = self.font_bold.render(state_str[:8], True, (255, 255, 255))
            # Center the tag label text
            tag_text_w = tag_lbl.get_width()
            screen.blit(tag_lbl, (tag_rect.x + (tag_rect.width - tag_text_w) // 2, tag_rect.y + 2))

            # Battery bar
            bat_bar_x = sb_x + 170
            bat_bar_w = sb_w - 170
            bat_bar_h = 10
            bat_ratio = max(0.0, min(1.0, d.battery / max(self.sim.config.battery_capacity, 1.0)))
            
            # Color coding battery
            if bat_ratio > 0.5:
                bat_color = (16, 185, 129)  # Green
            elif bat_ratio > 0.2:
                bat_color = (245, 158, 11)   # Yellow
            else:
                bat_color = (239, 68, 68)    # Red

            pygame.draw.rect(screen, (50, 50, 50), (bat_bar_x, item_y + 5, bat_bar_w, bat_bar_h), border_radius=3)
            if d.alive:
                pygame.draw.rect(screen, bat_color, (bat_bar_x, item_y + 5, int(bat_bar_w * bat_ratio), bat_bar_h), border_radius=3)
                # Percent text
                pct_str = f"{int(bat_ratio * 100)}%"
                pct_lbl = self.font_mono.render(pct_str, True, self.colors["text_muted"])
                screen.blit(pct_lbl, (bat_bar_x + bat_bar_w - pct_lbl.get_width() - 2, item_y + 4))
            else:
                # Dead indicator text
                dead_lbl = self.font_bold.render("CRASHED", True, self.colors["dead"])
                screen.blit(dead_lbl, (bat_bar_x + 2, item_y + 2))

            item_y += row_h

        pygame.display.flip()

    def shutdown(self):
        """Cleans up Pygame resources and closes the display window."""
        pygame.quit()
